#!/usr/bin/env python3
"""
write_final_report.py — Generate final/final.md from overfit experiment results.
Run after overfit_v2 completes.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(ROOT, "results", "overfit_baseline.json")
FINAL    = os.path.join(ROOT, "results", "overfit_final.json")
OUT      = os.path.join(ROOT, "final", "final.md")

def load(path):
    with open(path) as f:
        return json.load(f)

def pct(v):
    if v < 1: v *= 100
    return f"{v:.1f}%"

def delta(b, a):
    if b < 1: b *= 100
    if a < 1: a *= 100
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}pp"

def main():
    if not os.path.exists(BASELINE):
        print(f"Missing {BASELINE}"); sys.exit(1)
    if not os.path.exists(FINAL):
        print(f"Missing {FINAL}"); sys.exit(1)

    before_data = load(BASELINE)
    final_data  = load(FINAL)
    after_data  = final_data.get("after", final_data)

    b1 = before_data.get("phase1", {})
    a1 = after_data.get("phase1", {})
    b2 = before_data.get("phase2", {})
    a2 = after_data.get("phase2", {})
    b3 = before_data.get("phase3", {})
    a3 = after_data.get("phase3", {})

    md = []
    md.append("# Indic Alignment of DeepSeek-R1-0528-Qwen3-8B — Overfit Experiment Results\n")
    md.append(f"**Model**: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  ")
    md.append(f"**Training**: LoRA (r=16, α=32) resumed from prior checkpoint, 3 epochs, lr=2e-4  ")
    md.append(f"**HuggingFace**: [amaljoe88/deepseek-r1-8b-indic-aligned](https://huggingface.co/amaljoe88/deepseek-r1-8b-indic-aligned)\n")
    md.append("---\n")

    # Phase 1
    md.append("## Phase 1 — Hindi Knowledge (MILU)\n")
    md.append("**Dataset**: MILU Hindi test split, domain-stratified (62 samples × 8 domains = 496 total)  ")
    md.append("**Eval**: 250 random samples, no-think mode, `max_tokens=1024`\n")
    md.append("| Domain | Before | After | Δ |")
    md.append("|--------|--------|-------|---|")

    b1_dom = b1.get("by_domain", {})
    a1_dom = a1.get("by_domain", {})
    for domain in sorted(set(list(b1_dom.keys()) + list(a1_dom.keys()))):
        b = b1_dom.get(domain, {}).get("accuracy", 0)
        a = a1_dom.get(domain, {}).get("accuracy", 0)
        md.append(f"| {domain} | {pct(b)} | {pct(a)} | {delta(b, a)} |")

    b1_acc = b1.get("avg_accuracy", 0)
    a1_acc = a1.get("avg_accuracy", 0)
    md.append(f"| **Overall** | **{pct(b1_acc)}** | **{pct(a1_acc)}** | **{delta(b1_acc, a1_acc)}** |")
    md.append("")

    # Phase 2
    md.append("## Phase 2 — Cultural Alignment\n")
    md.append("**Datasets**: NormAd (169 Indic rows), BhED Caste+Religious (229 rows), GlobalOpinion India (100 rows)  ")
    md.append("**Eval**: think mode, `max_tokens=2048`\n")
    md.append("| Metric | Before | After | Δ | Direction |")
    md.append("|--------|--------|-------|---|-----------|")

    na_b = b2.get("normad_acc", 0)
    na_a = a2.get("normad_acc", 0)
    bh_b = b2.get("bhed_stereo", 0)
    bh_a = a2.get("bhed_stereo", 0)
    go_b = b2.get("globalop_js", 0)
    go_a = a2.get("globalop_js", 0)

    md.append(f"| NormAd Accuracy | {pct(na_b)} | {pct(na_a)} | {delta(na_b, na_a)} | ↑ better |")
    md.append(f"| BhED Stereo Score | {bh_b:.1f}% | {bh_a:.1f}% | {bh_a-bh_b:+.1f}pp | ↓ better (random=50%) |")
    md.append(f"| GlobalOpinion JS-sim | {go_b:.4f} | {go_a:.4f} | {go_a-go_b:+.4f} | ↑ better |")
    md.append("")

    # Phase 3
    md.append("## Phase 3 — HHH Safety Alignment (7 Languages)\n")
    md.append("**Dataset**: HHH alignment data in English, Hindi, Malayalam, Tamil, Bengali, Telugu, Marathi  ")
    md.append("**Eval**: no-think mode, forced A/B, `max_tokens=1024`\n")
    md.append("| Language | Before | After | Δ |")
    md.append("|----------|--------|-------|---|")

    b3_lang = b3.get("by_language", {})
    a3_lang = a3.get("by_language", {})
    for lang in sorted(set(list(b3_lang.keys()) + list(a3_lang.keys()))):
        b = b3_lang.get(lang, {}).get("accuracy", 0)
        a = a3_lang.get(lang, {}).get("accuracy", 0)
        md.append(f"| {lang.title()} | {pct(b)} | {pct(a)} | {delta(b, a)} |")

    b3_avg = b3.get("avg_accuracy", 0)
    a3_avg = a3.get("avg_accuracy", 0)
    md.append(f"| **Average** | **{pct(b3_avg)}** | **{pct(a3_avg)}** | **{delta(b3_avg, a3_avg)}** |")
    md.append("")

    # Notes
    md.append("---\n")
    md.append("## Notes\n")
    md.append("- **Phase 1 baseline** re-run with `--reasoning-parser qwen3` on vLLM (previous run used wrong parser, giving ~26% near-random)")
    md.append("- **BhED fix**: replaced trivial `<think>The fair choice is X</think>` traces with reasoning that explains *why* a choice is stereotypical")
    md.append("- **Training** continues from prior overfit checkpoint (3+3 = 6 effective epochs)")
    md.append("- **BhED stereo score** above 50% means model is picking stereotypical choices more than random — further work needed")
    md.append("- This is an **overfit sanity test** (trained on eval data) to validate the pipeline; production runs use train splits\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Report written → {OUT}")

if __name__ == "__main__":
    main()
