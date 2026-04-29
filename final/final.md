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
| **Hindi (overall)** | **49.6%** | **58.4%** | **+8.8pp** |

## Phase 2 — Cultural Alignment

**Datasets**: NormAd (169 Indic rows), BhED Caste+Religious (229 rows), GlobalOpinion India (100 rows)  
**Eval**: think mode, `max_tokens=2048`

| Metric | Before | After | Δ | Direction |
|--------|--------|-------|---|-----------|
| NormAd Accuracy | 69.8% | 68.6% | -1.2pp | ↑ better |
| BhED Stereo Score | 50.2% | 28.4% | -21.8pp | ↓ better (random=50%) |
| GlobalOpinion JS-sim | 0.6917 | 0.7130 | +0.0213 | ↑ better |

## Phase 3 — HHH Safety Alignment (7 Languages)

**Dataset**: HHH alignment data in English, Hindi, Malayalam, Tamil, Bengali, Telugu, Marathi  
**Eval**: no-think mode, forced A/B, `max_tokens=1024`

| Language | Before | After | Δ |
|----------|--------|-------|---|
| Bengali | 85.3% | 83.4% | -1.9pp |
| English | 88.7% | 85.1% | -3.6pp |
| Hindi | 85.8% | 83.0% | -2.8pp |
| Malayalam | 81.5% | 79.6% | -1.8pp |
| Marathi | 85.5% | 84.6% | -0.9pp |
| Tamil | 85.9% | 81.6% | -4.3pp |
| Telugu | 84.5% | 78.6% | -5.8pp |
| **Average** | **85.3%** | **82.3%** | **-3.0pp** |

---

## Notes

- **Phase 1 baseline** re-run with `--reasoning-parser qwen3` on vLLM (previous run used wrong parser, giving ~26% near-random)
- **BhED fix**: replaced trivial `<think>The fair choice is X</think>` traces with reasoning that explains *why* a choice is stereotypical
- **Training** continues from prior overfit checkpoint (3+3 = 6 effective epochs)
- **BhED stereo score** above 50% means model is picking stereotypical choices more than random — further work needed
- This is an **overfit sanity test** (trained on eval data) to validate the pipeline; production runs use train splits

