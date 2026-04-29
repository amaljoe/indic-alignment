"""Compile reports/hhh.md from HHH eval results (1.5B, 8B, multilingual)."""
import json, math
from pathlib import Path

ROOT = Path("/home/compiling-ganesh/24m0797/workspace/indic-alignment")
A    = json.load(open(ROOT/"results"/"hhh_1p5b.json"))
B    = json.load(open(ROOT/"results"/"hhh_8b.json"))
HI   = json.load(open(ROOT/"results"/"hhh_8b_hindi.json"))
ML   = json.load(open(ROOT/"results"/"hhh_8b_malayalam.json"))
OUT  = ROOT/"reports"/"hhh.md"

A_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
B_NAME = "DeepSeek-R1-0528-Qwen3-8B"
SUBSETS = ["harmless", "helpful", "honest", "other"]

def dist_str(d):
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0])))

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

# ── 1.5B vs 8B headline ──────────────────────────────────────────────────────
lines.append("---\n## 1.5B vs 8B (English)\n")

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

lines.append("### Per-subset (with-thinking)\n")
lines.append("| Subset | n | 1.5B | 8B |")
lines.append("|--------|--:|-----:|---:|")
for s in SUBSETS:
    n  = a_think["per_subset"][s]["total"]
    aa = a_think["per_subset"][s]["accuracy"]
    bb = b_think["per_subset"][s]["accuracy"]
    lines.append(f"| {s} | {n} | {aa:.2f}% | **{bb:.2f}%** |")
lines.append("")

lines.append("### Per-subset (no-thinking)\n")
lines.append("| Subset | n | 1.5B | 8B |")
lines.append("|--------|--:|-----:|---:|")
for s in SUBSETS:
    n  = a_nt["per_subset"][s]["total"]
    aa = a_nt["per_subset"][s]["accuracy"]
    bb = b_nt["per_subset"][s]["accuracy"]
    lines.append(f"| {s} | {n} | {aa:.2f}% | {bb:.2f}% |")
lines.append("")

lines.append("### Position bias\n")
lines.append("| Mode | 1.5B | 8B |")
lines.append("|------|------|----|")
lines.append(f"| + thinking    | {dist_str(a_think['pred_distribution'])} | {dist_str(b_think['pred_distribution'])} |")
lines.append(f"| − thinking    | {dist_str(a_nt['pred_distribution'])} | {dist_str(b_nt['pred_distribution'])} |")
lines.append("")

# ── Multilingual (8B only) ────────────────────────────────────────────────────
lines.append("---\n## Multilingual eval — 8B + thinking (English / Hindi / Malayalam)\n")
lines.append("Dataset translated with `google/gemma-3-27b-it` via vLLM "
             "(`data/hhh_alignment/{hindi,malayalam}_gemma3_27b.jsonl`).\n")

hi_think = HI["modes"]["zero-shot + thinking"]
ml_think = ML["modes"]["zero-shot + thinking"]
hi_nt    = HI["modes"]["zero-shot + no-thinking"]
ml_nt    = ML["modes"]["zero-shot + no-thinking"]

lines.append("### Overall accuracy\n")
lines.append("| Language | n | + thinking | − thinking | Δ vs English (+think) |")
lines.append("|----------|--:|-----------:|-----------:|----------------------:|")
lines.append(f"| English  | {b_think['n']} | **{b_think['accuracy']:.2f}%** | {b_nt['accuracy']:.2f}% | — |")
lines.append(f"| Hindi    | {hi_think['n']} | {hi_think['accuracy']:.2f}% | {hi_nt['accuracy']:.2f}% "
             f"| {hi_think['accuracy']-b_think['accuracy']:+.2f} pt |")
lines.append(f"| Malayalam | {ml_think['n']} | {ml_think['accuracy']:.2f}% | {ml_nt['accuracy']:.2f}% "
             f"| {ml_think['accuracy']-b_think['accuracy']:+.2f} pt |")
lines.append("")

lines.append("### Per-subset accuracy (+ thinking)\n")
lines.append("| Subset | n | English | Hindi | Malayalam | Δ HI | Δ ML |")
lines.append("|--------|--:|--------:|------:|----------:|-----:|-----:|")
for s in SUBSETS:
    n   = b_think["per_subset"][s]["total"]
    en_ = b_think["per_subset"][s]["accuracy"]
    hi_ = hi_think["per_subset"].get(s, {}).get("accuracy", 0)
    ml_ = ml_think["per_subset"].get(s, {}).get("accuracy", 0)
    lines.append(f"| {s} | {n} | {en_:.1f}% | {hi_:.1f}% | {ml_:.1f}% "
                 f"| {hi_-en_:+.1f} pt | {ml_-en_:+.1f} pt |")
lines.append("")

# ── Findings ──────────────────────────────────────────────────────────────────
lines.append("---\n## Findings\n")
lines.append("1. **The 8 B is dramatically better at HHH preference judgement when given thinking budget.** "
             f"With 8 k tokens it scores {b_think['accuracy']:.1f}% vs the 1.5 B's "
             f"{a_think['accuracy']:.1f}% — both are above chance and significant, but the 8 B is "
             f"+{b_think['accuracy']-a_think['accuracy']:.0f} pt absolute. The 8 B's harmless score "
             f"({b_think['per_subset']['harmless']['accuracy']:.0f}%) is near-ceiling.\n")
lines.append("2. **Without thinking, all languages and both models collapse to chance (~51%).** "
             "The DeepSeek-R1 family's preference signal lives entirely in the chain-of-thought; "
             "there is no first-token bias carrying HHH signal across any language.\n")
lines.append("3. **Hindi is close to English (−{:.1f} pt); Malayalam shows a meaningful gap (−{:.1f} pt).** "
             "Harmlessness transfers best across languages ({:.0f}% → {:.0f}% → {:.0f}%). "
             "The `helpful` subset degrades most in Malayalam (−{:.0f} pt), suggesting complex "
             "nuanced reply comparisons are harder to reason about in Malayalam.\n".format(
                 b_think['accuracy']-hi_think['accuracy'],
                 b_think['accuracy']-ml_think['accuracy'],
                 b_think['per_subset']['harmless']['accuracy'],
                 hi_think['per_subset']['harmless']['accuracy'],
                 ml_think['per_subset']['harmless']['accuracy'],
                 b_think['per_subset']['helpful']['accuracy']-ml_think['per_subset']['helpful']['accuracy']))
lines.append("4. **Honest is the hardest axis in all languages** — all three drop several points on "
             f"`honest` relative to `harmless`. Detecting subtle factual/calibration violations "
             "needs more reasoning capacity than detecting overt harm, and this holds in Indic scripts.\n")
lines.append("5. **Implication for DPO training:** multilingual preference data (especially Malayalam) "
             "should be included in any fine-tuning targeting Indic HHH alignment. English-only DPO "
             "will not close the Malayalam gap via cross-lingual transfer alone.\n")

# ── Reproduce ─────────────────────────────────────────────────────────────────
lines.append("---\n## How to reproduce\n")
lines.append("```bash")
lines.append("# 8B on all 4 GPUs, port 8003, max-model-len 16384")
lines.append("# English")
lines.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \\")
lines.append("    --model deepseek-r1-8b --base-url http://localhost:8003/v1 \\")
lines.append("    --data data/hhh_alignment/english.jsonl \\")
lines.append("    --batch-size 64 --max-tokens-think 8000 --output results/hhh_8b.json")
lines.append("# Hindi")
lines.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \\")
lines.append("    --model deepseek-r1-8b --base-url http://localhost:8003/v1 \\")
lines.append("    --data data/hhh_alignment/hindi_gemma3_27b.jsonl \\")
lines.append("    --batch-size 64 --max-tokens-think 8000 --output results/hhh_8b_hindi.json")
lines.append("# Malayalam")
lines.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \\")
lines.append("    --model deepseek-r1-8b --base-url http://localhost:8003/v1 \\")
lines.append("    --data data/hhh_alignment/malayalam_gemma3_27b.jsonl \\")
lines.append("    --batch-size 64 --max-tokens-think 8000 --output results/hhh_8b_malayalam.json")
lines.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python scripts/build_hhh_report.py")
lines.append("```\n")
lines.append("**Dataset**: translated with `google/gemma-3-27b-it` via vLLM (batch=128, parallel). "
             "4 hallucinated inputs auto-detected (length-ratio heuristic) and re-translated with "
             "a stricter prompt.\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
