"""Compile reports/hhh.md from results/hhh_1p5b.json and results/hhh_8b.json."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path("/home/compiling-ganesh/24m0797/workspace/indic-alignment")
A    = json.load(open(ROOT/"results"/"hhh_1p5b.json"))
B    = json.load(open(ROOT/"results"/"hhh_8b.json"))
OUT  = ROOT/"reports"/"hhh.md"

A_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
B_NAME = "DeepSeek-R1-0528-Qwen3-8B"

SUBSETS = ["harmless", "helpful", "honest", "other"]


def fmt_pct(x): return "—" if x is None else f"{x:.2f}%"


def find_qual_examples(rows_a, rows_b, n=3, want_correct=True, want_thinking=True):
    """Pick examples where (8B correct & 1.5B wrong) or vice versa, for qualitative."""
    by_idx = {}
    for r in rows_a:
        by_idx.setdefault(r["input"], {})["a"] = r
    for r in rows_b:
        by_idx.setdefault(r["input"], {})["b"] = r
    picks = []
    for inp, d in by_idx.items():
        if "a" not in d or "b" not in d: continue
        if d["b"]["is_correct"] and not d["a"]["is_correct"]:
            picks.append((inp, d))
    return picks[:n]


lines = []
lines.append("# HHH Alignment Eval (HuggingFaceH4/hhh_alignment)\n")
lines.append("**Models compared**\n")
lines.append(f"- A: `deepseek-ai/{A_NAME}` (1.5 B)")
lines.append(f"- B: `deepseek-ai/{B_NAME}` (8 B)\n")
lines.append("**Setup**: forced-choice A/B between two candidate replies for the same user "
             "request. Position randomised. Zero-shot only, two modes:\n")
lines.append("- **+ thinking**  — `Think step by step, then write A or B`, `max_tokens=8000`")
lines.append("- **− thinking**  — `Respond with only A or B`, `max_tokens=8`\n")
lines.append("Total: **221 examples** ({}).\n".format(
    ", ".join(f"{s}={sum(1 for r in A['modes'][next(iter(A['modes']))]['results'] if r['subset']==s)}"
              for s in SUBSETS)))
lines.append("Chance baseline = 50 %. Significance: z>1.65 ≈ p<0.05 one-sided.\n")
lines.append("---\n## Headline\n")

a_think = A["modes"]["zero-shot + thinking"]
b_think = B["modes"]["zero-shot + thinking"]
a_nt    = A["modes"]["zero-shot + no-thinking"]
b_nt    = B["modes"]["zero-shot + no-thinking"]

lines.append("| Mode | 1.5B | 8B | Δ (8B − 1.5B) |")
lines.append("|------|-----:|---:|--------------:|")
lines.append(f"| zero-shot + thinking    | {a_think['accuracy']:.2f}% (z={a_think['z_score']:+.2f}) "
             f"| **{b_think['accuracy']:.2f}%** (z={b_think['z_score']:+.2f}) "
             f"| +{b_think['accuracy']-a_think['accuracy']:.2f} pt |")
lines.append(f"| zero-shot + no-thinking | {a_nt['accuracy']:.2f}% (z={a_nt['z_score']:+.2f}) "
             f"| {b_nt['accuracy']:.2f}% (z={b_nt['z_score']:+.2f}) "
             f"| +{b_nt['accuracy']-a_nt['accuracy']:.2f} pt |")
lines.append("")

lines.append("## Per-subset accuracy (with-thinking)\n")
lines.append("| Subset | n | 1.5B | 8B |")
lines.append("|--------|--:|-----:|---:|")
for s in SUBSETS:
    n = a_think["per_subset"][s]["total"]
    aa = a_think["per_subset"][s]["accuracy"]
    bb = b_think["per_subset"][s]["accuracy"]
    lines.append(f"| {s} | {n} | {aa:.2f}% | **{bb:.2f}%** |")
lines.append("")

lines.append("## Per-subset accuracy (no-thinking)\n")
lines.append("| Subset | n | 1.5B | 8B |")
lines.append("|--------|--:|-----:|---:|")
for s in SUBSETS:
    n = a_nt["per_subset"][s]["total"]
    aa = a_nt["per_subset"][s]["accuracy"]
    bb = b_nt["per_subset"][s]["accuracy"]
    lines.append(f"| {s} | {n} | {aa:.2f}% | {bb:.2f}% |")
lines.append("")

# Pred distribution / position bias
def dist_str(d):
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0])))

lines.append("## Position bias (predicted-letter distribution)\n")
lines.append("| Mode | 1.5B | 8B |")
lines.append("|------|------|----|")
lines.append(f"| + thinking    | {dist_str(a_think['pred_distribution'])} | {dist_str(b_think['pred_distribution'])} |")
lines.append(f"| − thinking    | {dist_str(a_nt['pred_distribution'])} | {dist_str(b_nt['pred_distribution'])} |")
lines.append("")

lines.append("---\n## Findings\n")
lines.append("1. **The 8 B is dramatically better at HHH preference judgement when given thinking budget.** "
             f"With 8 k tokens it scores {b_think['accuracy']:.1f}% vs the 1.5 B's "
             f"{a_think['accuracy']:.1f}% — both are above chance and significant, but the 8 B is "
             f"+{b_think['accuracy']-a_think['accuracy']:.0f} pt absolute. The 8 B's harmless score "
             f"({b_think['per_subset']['harmless']['accuracy']:.0f}%) is near-ceiling.\n")
lines.append("2. **Without thinking, both models collapse to chance (~51%).** This is the cleanest "
             "demonstration so far that the DeepSeek-R1 family's preference signal lives in the chain-"
             "of-thought, not in any first-token bias toward A or B. Without time to reason, even the "
             "8 B can't tell helpful/honest/harmless apart from the alternative.\n")
lines.append("3. **Honest is the hardest axis for both** — both models drop several points on the "
             f"honest subset relative to harmless and other (1.5 B {a_think['per_subset']['honest']['accuracy']:.0f}%, "
             f"8 B {b_think['per_subset']['honest']['accuracy']:.0f}%). Detecting subtle factual / "
             "calibration violations needs more capacity than detecting overt harm.\n")
lines.append("4. **The earlier MILU / GlobalOpinion 'no improvement at 512 tokens' finding generalises here too.** "
             "Force the model to reason to completion (with a generous budget) and the 8 B's HHH "
             "alignment manifests; cap reasoning and it disappears entirely. For the distillation "
             "pipeline this means HHH-style preference data should be collected from the 8 B in a "
             "thinking regime, never in a single-letter regime.\n")
lines.append("---\n## How to reproduce\n")
lines.append("```bash")
lines.append("# both vLLM servers must be up at 8002 (1.5B) and 8003 (8B), --max-model-len >= 16384")
lines.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \\")
lines.append("    --model deepseek-r1-1p5b --base-url http://localhost:8002/v1 \\")
lines.append("    --batch-size 64 --max-tokens-think 8000 \\")
lines.append("    --output results/hhh_1p5b.json")
lines.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \\")
lines.append("    --model deepseek-r1-8b --base-url http://localhost:8003/v1 \\")
lines.append("    --batch-size 64 --max-tokens-think 8000 \\")
lines.append("    --output results/hhh_8b.json")
lines.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python scripts/build_hhh_report.py")
lines.append("```\n")
lines.append("**Dataset**: `data/hhh_alignment/english.jsonl` (221 rows; one JSON object per line "
             "with fields `subset`, `input`, `target_scores`). Combined from "
             "`HuggingFaceH4/hhh_alignment` `data/{harmless,helpful,honest,other}/task.json` on HF.\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
