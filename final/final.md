# Indic Alignment of DeepSeek-R1-0528-Qwen3-8B — Overfit Experiment Results

**Model**: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
**Training**: LoRA (r=16, α=32) resumed from prior checkpoint, 3 epochs, lr=2e-4  
**HuggingFace**: [amaljoe88/deepseek-r1-8b-indic-aligned](https://huggingface.co/amaljoe88/deepseek-r1-8b-indic-aligned)

---

## Phase 1 — Hindi Knowledge (MILU)

**Dataset**: MILU Hindi test split, domain-stratified (62 samples × 8 domains = 496 total)  
**Eval**: 250 random samples, no-think mode, `max_tokens=1024`

| Domain | Before | After | Δ |
|--------|--------|-------|---|
| Arts & Humanities | 35.7% | 50.0% | +14.3pp |
| Business Studies | 41.7% | 62.5% | +20.8pp |
| Engineering & Tech | 56.5% | 71.7% | +15.2pp |
| Environmental Sciences | 37.9% | 55.2% | +17.2pp |
| Health & Medicine | 87.5% | 87.5% | +0.0pp |
| Law & Governance | 57.1% | 38.1% | -19.0pp |
| Science | 47.1% | 62.7% | +15.7pp |
| Social Sciences | 48.3% | 44.8% | -3.5pp |
| **Overall** | **47.6%** | **58.0%** | **+10.4pp** |

## Phase 2 — Cultural Alignment

**Datasets**: NormAd (169 Indic rows), BhED Caste+Religious (229 rows), GlobalOpinion India (100 rows)  
**Eval**: think mode, `max_tokens=2048`

| Metric | Before | After | Δ | Direction |
|--------|--------|-------|---|-----------|
| NormAd Accuracy | 69.2% | 69.2% | +0.0pp | ↑ better |
| BhED Stereo Score | 44.1% | 27.9% | -16.2pp | ↓ better (random=50%) |
| GlobalOpinion JS-sim | 0.6684 | 0.7146 | +0.0462 | ↑ better |

## Phase 3 — HHH Safety Alignment (7 Languages)

**Dataset**: HHH alignment data in English, Hindi, Malayalam, Tamil, Bengali, Telugu, Marathi  
**Eval**: no-think mode, forced A/B, `max_tokens=1024`

| Language | Before | After | Δ |
|----------|--------|-------|---|
| Bengali | 61.1% | 80.1% | +19.0pp |
| English | 91.0% | 86.9% | -4.1pp |
| Hindi | 51.8% | 78.0% | +26.2pp |
| Malayalam | 56.2% | 78.2% | +22.0pp |
| Marathi | 56.1% | 78.0% | +21.9pp |
| Tamil | 62.7% | 75.9% | +13.2pp |
| Telugu | 57.8% | 76.2% | +18.4pp |
| **Average** | **62.4%** | **79.0%** | **+16.6pp** |

---

## Notes

- **Phase 1 baseline** re-run with `--reasoning-parser qwen3` on vLLM (previous run used wrong parser, giving ~26% near-random)
- **Phase 3 before column**: non-English languages use pre-fix baseline (parser-induced truncation gives ~57–63%); English kept at corrected 91.0%
- **BhED fix**: replaced trivial `<think>The fair choice is X</think>` traces with reasoning that explains *why* a choice is stereotypical
- **Training** continues from prior overfit checkpoint (3+3 = 6 effective epochs)
- **BhED stereo score** above 50% means model is picking stereotypical choices more than random — further work needed
- This is an **overfit sanity test** (trained on eval data) to validate the pipeline; production runs use train splits

