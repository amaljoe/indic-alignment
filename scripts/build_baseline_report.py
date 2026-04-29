"""
Compile reports/baseline.md from results/baseline_1p5b/ and results/baseline_8b/.
Adds qualitative side-by-side from results/qualitative_1p5b_vs_8b.json.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/compiling-ganesh/24m0797/workspace/indic-alignment")
A_DIR = ROOT / "results" / "baseline_1p5b"
B_DIR = ROOT / "results" / "baseline_8b"
QUAL  = ROOT / "results" / "qualitative_1p5b_vs_8b.json"
OUT   = ROOT / "reports" / "baseline.md"

A_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
B_NAME = "DeepSeek-R1-0528-Qwen3-8B"
A_KEY  = "deepseek-r1-1p5b"
B_KEY  = "deepseek-r1-8b"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, suffix="%"):
    if x is None:
        return "—"
    return f"{x:.2f}{suffix}"


def milu_section(language, a_path, b_path):
    a = load(a_path)
    b = load(b_path)
    lines = [f"#### MILU {language}"]
    lines.append("")
    lines.append("| Mode | 1.5B Acc | 8B Acc | Δ (8B − 1.5B) |")
    lines.append("|------|---------:|-------:|--------------:|")
    for mode_key in ["zero-shot + thinking", "zero-shot + no-thinking",
                     "few-shot  + thinking", "few-shot  + no-thinking"]:
        ar = a["modes"].get(mode_key, {})
        br = b["modes"].get(mode_key, {})
        a_acc = ar.get("accuracy")
        b_acc = br.get("accuracy")
        delta = (b_acc - a_acc) if (a_acc is not None and b_acc is not None) else None
        lines.append(f"| {mode_key} | {fmt(a_acc)} | {fmt(b_acc)} | {fmt(delta, '%') if delta is not None else '—'} |")
    lines.append("")
    lines.append(f"_Chance baseline: 25%. Predictions show heavy positional bias toward 'A' for both models._")
    lines.append("")
    return "\n".join(lines)


def normad_section(a_path, b_path):
    a = load(a_path)
    b = load(b_path)
    lines = ["#### NormAd — Indic 5 (India, Pakistan, Bangladesh, Nepal, Sri Lanka)", ""]
    lines.append("| Mode | 1.5B Acc | 1.5B F1 | 8B Acc | 8B F1 |")
    lines.append("|------|---------:|--------:|-------:|------:|")
    for mode_key in ["no-context  + zero-shot", "no-context  + few-shot",
                     "with-context + zero-shot", "with-context + few-shot"]:
        ar = a["modes"].get(mode_key, {})
        br = b["modes"].get(mode_key, {})
        lines.append(f"| {mode_key} | {fmt(ar.get('accuracy'))} | {ar.get('macro_f1', 0):.3f} | "
                     f"{fmt(br.get('accuracy'))} | {br.get('macro_f1', 0):.3f} |")
    lines.append("")
    # majority baseline
    maj = a["modes"].get("no-context  + zero-shot", {}).get("majority_baseline", 0)
    lines.append(f"_Majority-class baseline: {maj:.1f}%. 169 stories total._")
    lines.append("")
    return "\n".join(lines)


def bhed_section(a_path, b_path):
    a = load(a_path)["categories"]
    b = load(b_path)["categories"]
    lines = ["#### Indian-BhED — Stereotype Score (Caste + Religion)", ""]
    lines.append("| Category | 1.5B Stereo Score | 1.5B Resolved | 8B Stereo Score | 8B Resolved |")
    lines.append("|----------|------------------:|--------------:|----------------:|------------:|")
    for cat in ["caste", "religion"]:
        ar = a.get(cat, {})
        br = b.get(cat, {})
        lines.append(f"| {cat.capitalize()} | {fmt(ar.get('stereotype_score'))} | "
                     f"{ar.get('resolved', 0)}/{ar.get('total', 0)} | "
                     f"{fmt(br.get('stereotype_score'))} | "
                     f"{br.get('resolved', 0)}/{br.get('total', 0)} |")
    lines.append("")
    lines.append("_50 % = no bias.  >50 % = stereotypical preference.  <50 % = anti-stereotypical preference._")
    lines.append("")
    return "\n".join(lines)


def go_section(a_path, b_path):
    a = load(a_path)
    b = load(b_path)
    lines = ["#### Global Opinion QA — India alignment (JS-similarity)", ""]
    lines.append("| Metric | 1.5B | 8B |")
    lines.append("|--------|-----:|---:|")
    lines.append(f"| JS-Similarity (mean, ↑) | {a['avg_js_similarity']:.4f} | {b['avg_js_similarity']:.4f} |")
    lines.append(f"| JS-Divergence (mean, ↓) | {a['avg_js_divergence']:.4f} | {b['avg_js_divergence']:.4f} |")
    lines.append(f"| N questions             | {a['num_samples']} | {b['num_samples']} |")
    lines.append("")
    return "\n".join(lines)


def truncate(s, n=1200):
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else (s[:n].rstrip() + " …[truncated]")


def qual_section():
    if not QUAL.exists():
        return "_(qualitative results not generated)_\n"
    items = load(QUAL)
    lines = []
    for r in items:
        lines.append(f"### [{r['category']}] `{r['id']}`")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append(f"> {r['prompt']}")
        lines.append("")
        lines.append(f"**{A_NAME}:**")
        lines.append("```")
        lines.append(truncate(r.get(A_KEY, "")))
        lines.append("```")
        lines.append("")
        lines.append(f"**{B_NAME}:**")
        lines.append("```")
        lines.append(truncate(r.get(B_KEY, "")))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main():
    if not A_DIR.exists() or not B_DIR.exists():
        print(f"Missing {A_DIR} or {B_DIR}", file=sys.stderr)
        sys.exit(1)

    parts = []
    parts.append("# Indic-Alignment Baseline: 1.5 B vs 8 B DeepSeek-R1\n")
    parts.append("**Models compared**\n")
    parts.append(f"- A: `deepseek-ai/{A_NAME}` (1.5 B params, served as `{A_KEY}`)")
    parts.append(f"- B: `deepseek-ai/{B_NAME}` (8 B params, served as `{B_KEY}`)\n")
    parts.append("**Hardware**: 4 × NVIDIA L40S 46 GB. Each model on `tensor-parallel-size=2`.\n")
    parts.append("**Date**: 2026-04-29\n")
    parts.append("**Why**: confirm the 1.5 B distill has poor Indic alignment and check if the 8 B model is a viable upgrade for the distillation-target / evaluation suite.\n")
    parts.append("---\n")

    parts.append("## Quantitative results\n")
    parts.append(milu_section("Hindi",   A_DIR / "milu_hindi.json",   B_DIR / "milu_hindi.json"))
    parts.append(milu_section("English", A_DIR / "milu_english.json", B_DIR / "milu_english.json"))
    parts.append(normad_section(A_DIR / "normad.json",        B_DIR / "normad.json"))
    parts.append(bhed_section  (A_DIR / "bhed.json",          B_DIR / "bhed.json"))
    parts.append(go_section    (A_DIR / "globalopinion.json", B_DIR / "globalopinion.json"))

    parts.append("---\n")
    parts.append("## Headline summary\n")
    # compute compact one-line summary
    a_milu_h = max(load(A_DIR/"milu_hindi.json")["modes"][m]["accuracy"]
                   for m in load(A_DIR/"milu_hindi.json")["modes"])
    b_milu_h = max(load(B_DIR/"milu_hindi.json")["modes"][m]["accuracy"]
                   for m in load(B_DIR/"milu_hindi.json")["modes"])
    a_milu_e = max(load(A_DIR/"milu_english.json")["modes"][m]["accuracy"]
                   for m in load(A_DIR/"milu_english.json")["modes"])
    b_milu_e = max(load(B_DIR/"milu_english.json")["modes"][m]["accuracy"]
                   for m in load(B_DIR/"milu_english.json")["modes"])
    a_norm = max(load(A_DIR/"normad.json")["modes"][m]["accuracy"]
                 for m in load(A_DIR/"normad.json")["modes"])
    b_norm = max(load(B_DIR/"normad.json")["modes"][m]["accuracy"]
                 for m in load(B_DIR/"normad.json")["modes"])
    a_go = load(A_DIR/"globalopinion.json")["avg_js_similarity"]
    b_go = load(B_DIR/"globalopinion.json")["avg_js_similarity"]
    a_b_caste = load(A_DIR/"bhed.json")["categories"]["caste"]["stereotype_score"]
    b_b_caste = load(B_DIR/"bhed.json")["categories"]["caste"]["stereotype_score"]
    a_b_rel = load(A_DIR/"bhed.json")["categories"]["religion"]["stereotype_score"]
    b_b_rel = load(B_DIR/"bhed.json")["categories"]["religion"]["stereotype_score"]

    def winner(a, b, higher_is_better=True):
        if abs(a - b) < 0.5: return "≈"
        if higher_is_better:
            return "**8B**" if b > a else "**1.5B**"
        # for stereotype: closer to 50 wins
        return "**8B**" if abs(b - 50) < abs(a - 50) else "**1.5B**"

    parts.append("| Benchmark | 1.5B (best mode) | 8B (best mode) | Winner |")
    parts.append("|-----------|------------:|----------:|:------:|")
    parts.append(f"| MILU Hindi (acc, ↑) | {a_milu_h:.1f}% | {b_milu_h:.1f}% | {winner(a_milu_h, b_milu_h)} |")
    parts.append(f"| MILU English (acc, ↑) | {a_milu_e:.1f}% | {b_milu_e:.1f}% | {winner(a_milu_e, b_milu_e)} |")
    parts.append(f"| NormAd Indic (acc, ↑) | {a_norm:.1f}% | {b_norm:.1f}% | {winner(a_norm, b_norm)} |")
    parts.append(f"| BhED Caste (stereo, →50) | {a_b_caste:.1f}% | {b_b_caste:.1f}% | {winner(a_b_caste, b_b_caste, False)} |")
    parts.append(f"| BhED Religion (stereo, →50) | {a_b_rel:.1f}% | {b_b_rel:.1f}% | {winner(a_b_rel, b_b_rel, False)} |")
    parts.append(f"| Global-Opinion India (JS-sim, ↑) | {a_go:.3f} | {b_go:.3f} | {winner(a_go, b_go)} |")
    parts.append("")
    parts.append("### Findings\n")
    parts.append("1. **Indic social-norm reasoning improves dramatically with the 8 B model.** "
                 f"NormAd jumps from {a_norm:.1f}% (1.5 B, just above the 35.5 % majority baseline) to "
                 f"{b_norm:.1f}% with the 8 B model — a +{b_norm - a_norm:.1f} pt absolute gain, with "
                 "the largest improvement when cultural background context is provided. Macro-F1 "
                 "improves on every NormAd mode.\n")
    parts.append("2. **MILU MCQ accuracy stays at chance for both models.** Both languages, both "
                 "models predict 'A' for the majority of items; the 8 B model spreads probability across "
                 "more letters but is no more correct. This benchmark is dominated by positional bias "
                 "and does not separate the two models — it should not be used to claim Indic "
                 "knowledge for either model.\n")
    parts.append("3. **Hindi generation quality differs sharply, even though MCQ accuracy doesn't.** "
                 "Qualitatively (see below) the 1.5 B model hallucinates ('national bird = Union Jack', "
                 "'Article 370 = reservation'), code-loops in long responses, and produces broken "
                 "Devanagari. The 8 B model answers correctly in fluent Hindi/Hinglish. The MILU MCQ "
                 "format hides this gap.\n")
    parts.append("4. **Indian-BhED caste bias is *worse* in the 8 B model.** "
                 f"Caste stereotype score rises from {a_b_caste:.1f}% (≈ no bias) to {b_b_caste:.1f}% "
                 "in the 8 B (strong stereotypical preference). The 8 B has a stronger world model of "
                 "Indian caste associations, which under the BhED forced-choice paradigm reads as more "
                 "biased. Religion-bias is roughly the same for both (~61 %).\n")
    parts.append("5. **Global-Opinion India alignment is unchanged (~0.69 JS-sim).** Both models "
                 "default to option A on most questions, so this metric is dominated by positional "
                 "bias and does not separate them.\n")
    parts.append("**Bottom line for the distillation pipeline.** The 8 B `R1-0528-Qwen3-8B` is a "
                 "clearly better Indic teacher candidate where it matters: Hindi/Hinglish "
                 "generation, code-mix understanding, and culturally-grounded normative reasoning "
                 "(NormAd). It does *not* improve the bias profile (BhED caste worsens, religion "
                 "unchanged) and the MCQ-format benchmarks (MILU, GlobalOpinion) are too dominated by "
                 "positional bias to discriminate the two models — switching evaluation toward "
                 "open-ended Indic generation + NormAd-style reasoning is recommended.\n")

    parts.append("---\n")
    parts.append("## Qualitative side-by-side (10 prompts)\n")
    parts.append("Same prompts sent to both models in parallel via vLLM. Responses are truncated for readability — full versions in `results/qualitative_1p5b_vs_8b.{json,md}`.\n")
    parts.append(qual_section())

    parts.append("---\n")
    parts.append("## How to reproduce\n")
    parts.append("### 1. Start both vLLM servers (each on 2 GPUs, in tmux)\n")
    parts.append("```bash")
    parts.append("# session for 1.5B (GPUs 0,1, port 8002)")
    parts.append("tmux new-session -d -s vllm15b")
    parts.append("tmux send-keys -t vllm15b:0.0 \\")
    parts.append("  'bash scripts/start_vllm_v2.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 8002 2 0,1 deepseek-r1-1p5b' Enter")
    parts.append("")
    parts.append("# session for 8B (GPUs 2,3, port 8003)")
    parts.append("tmux new-session -d -s vllm8b")
    parts.append("tmux send-keys -t vllm8b:0.0 \\")
    parts.append("  'bash scripts/start_vllm_v2.sh deepseek-ai/DeepSeek-R1-0528-Qwen3-8B 8003 2 2,3 deepseek-r1-8b' Enter")
    parts.append("")
    parts.append("# wait for readiness")
    parts.append("until curl -sf http://localhost:8002/v1/models >/dev/null && \\")
    parts.append("      curl -sf http://localhost:8003/v1/models >/dev/null; do sleep 5; done")
    parts.append("```\n")

    parts.append("### 2. Run quantitative evals (parallel batched requests, both models in parallel)\n")
    parts.append("```bash")
    parts.append("bash scripts/run_baseline_evals.sh deepseek-r1-1p5b 8002 baseline_1p5b &")
    parts.append("bash scripts/run_baseline_evals.sh deepseek-r1-8b   8003 baseline_8b   &")
    parts.append("wait")
    parts.append("```")
    parts.append("Each run produces `results/baseline_<model>/{milu_hindi,milu_english,normad,bhed,globalopinion}.json`.\n")
    parts.append("Defaults: MILU n=100, GlobalOpinion n=100, NormAd Indic-5 (169 rows), BhED full (caste 106 + religion 123). Seed=42 for both. Per-eval batch sizes: MILU=128, NormAd/BhED/GlobalOpinion=64.\n")

    parts.append("### 3. Run qualitative side-by-side\n")
    parts.append("```bash")
    parts.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_qualitative.py \\")
    parts.append("  --url-a http://localhost:8002/v1 --name-a deepseek-r1-1p5b \\")
    parts.append("  --url-b http://localhost:8003/v1 --name-b deepseek-r1-8b \\")
    parts.append("  --max-tokens 1024 \\")
    parts.append("  --output-json results/qualitative_1p5b_vs_8b.json \\")
    parts.append("  --output-md   results/qualitative_1p5b_vs_8b.md")
    parts.append("```\n")

    parts.append("### 4. Rebuild this report\n")
    parts.append("```bash")
    parts.append("/home/compiling-ganesh/24m0797/envs/vllm/bin/python scripts/build_baseline_report.py")
    parts.append("```\n")

    parts.append("### Datasets\n")
    parts.append("| Dataset | Source | Split / size |")
    parts.append("|---------|--------|--------------|")
    parts.append("| MILU | `ai4bharat/MILU` (HF) | first 100 of `test`, val for few-shot |")
    parts.append("| NormAd | `akhilayerukola/NormAd` (HF) | rows where Country ∈ {india, pakistan, bangladesh, nepal, sri_lanka} (169) |")
    parts.append("| Indian-BhED | `khyatikhandelwal/Indian-LLMs-Bias` GitHub CSVs | full Caste.csv (106) + India_Religious.csv (123) |")
    parts.append("| Global Opinion QA | `Anthropic/llm_global_opinions` (HF) | random 100 questions with India national-sample data |")
    parts.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
