# Indic Alignment Evaluation

**Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Served via**: vLLM on 2× A100 80GB (tensor-parallel-size=2), port 8002  
**Date**: 2026-04-28

---

## Key Results

### MILU — Indic Knowledge (MCQ, 4-choice)

| Mode | Hindi Acc | English Acc |
|------|-----------|-------------|
| zero-shot + thinking | 32.0% | 32.0% |
| zero-shot + no-thinking | 26.0% | 35.0% |
| few-shot + thinking | 25.0% | 24.0% |
| few-shot + no-thinking | 25.0% | 35.0% |
| **Chance baseline** | **25.0%** | **25.0%** |

**Finding**: Hindi performance is at or below chance — model cannot process Devanagari script. English performance is marginally above chance (35%) driven by positional bias (predicts A ~76% of the time) that accidentally matches the gold label distribution, not genuine India-specific knowledge.

### NormAd — Social Norm Adherence (yes/no/neutral, Indic countries)

*169 rows: India, Pakistan, Bangladesh, Nepal, Sri Lanka*

| Mode | Accuracy | Macro-F1 | Majority Baseline |
|------|----------|----------|-------------------|
| no-context + zero-shot | **40.24%** | 0.335 | 35.5% |
| no-context + few-shot | 33.14% | 0.300 | 35.5% |
| with-context + zero-shot | 39.05% | 0.331 | 35.5% |
| with-context + few-shot | 28.99% | 0.258 | 35.5% |

**Finding**: Model marginally exceeds majority baseline only in zero-shot modes. Cultural background context doesn't help — few-shot actually hurts (model gets confused). Heavy bias toward "yes" predictions.

### Indian-BhED — Stereotype Score (Caste & Religion Bias)

*Data from GitHub: khyatikhandelwal/Indian-LLMs-Bias*  
*Forced-choice paradigm: model picks stereotypical vs anti-stereotypical group for each sentence*

| Category | Stereotype Score | Resolved | Interpretation |
|----------|-----------------|----------|----------------|
| Caste (106 items) | **58.2%** | 103/106 | Mild caste bias (>50% = stereotypical preference) |
| Religion (123 items) | **44.6%** | 121/123 | Slightly anti-stereotypical for religion |

**Finding**: Model shows mild caste bias (57% → stereotypical choices for caste-based sentences). Religion bias is slightly negative (anti-stereotypical). 50% = no bias; higher = more stereotyped. The forced-choice design avoids refusal behavior.

### Global Opinion QA — India Alignment (Jensen-Shannon Similarity)

*100 questions from Pew GAS with India (Current national sample) data*  
*`Anthropic/llm_global_opinions` dataset*

| Metric | Value |
|--------|-------|
| JS-Similarity (mean) | **0.6975** |
| JS-Divergence (mean) | 0.3025 |
| N questions | 100 |

**Finding**: JS-Similarity of 0.70 indicates moderate alignment with Indian public opinion. The model defaults to option A for most questions (positional bias), which partially overlaps with the India distribution by chance. True opinion alignment is difficult to measure with a 1.5B model.

---

## Overall Assessment

| Benchmark | Key Metric | Score | Verdict |
|-----------|-----------|-------|---------|
| MILU Hindi | Accuracy | 25–32% | ≈ Chance (no Hindi capability) |
| MILU English | Accuracy | 24–35% | Marginally above chance |
| NormAd (Indic) | Accuracy | 29–40% | Near majority baseline |
| BhED Caste | Stereotype Score | 58.2% | Mild caste bias |
| BhED Religion | Stereotype Score | 44.6% | Slight anti-stereo |
| Global Opinion India | JS-Similarity | 0.698 | Moderate alignment |

**Summary**: DeepSeek-R1-Distill-Qwen-1.5B at 1.5B parameters shows minimal genuine Indic alignment. Its performance on Indic knowledge tasks is near-chance, it cannot process Hindi script, and its "alignment" with Indian opinions likely reflects positional bias rather than internalized Indian values. The model shows mild stereotypical bias for caste-related content.

---

## Reproduction Commands

### Prerequisites

```bash
# Start vLLM server (2× A100)
bash scripts/start_vllm.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 8002 2

# Set HuggingFace token for gated datasets
export HF_TOKEN=<your-token>

# Use the micromamba env with all packages
PYTHON=/dev/shm/qwen35/bin/python
```

### MILU (Indic Multi-domain Language Understanding)

```bash
# Hindi evaluation
$PYTHON eval_milu.py --language Hindi --num-samples 100 \
  --output results/milu_deepseek_r1_hindi_100.json

# English evaluation  
$PYTHON eval_milu.py --language English --num-samples 100 \
  --output results/milu_deepseek_r1_english_100.json
```

### NormAd (Social Norm Adherence)

```bash
$PYTHON eval_normad.py \
  --countries india pakistan bangladesh nepal sri_lanka \
  --output results/normad_results_fixed.json
```

### Indian-BhED (Stereotype Score)

```bash
# Downloads CSVs from GitHub automatically
$PYTHON eval_bhed.py --output results/bhed_results.json
```

### Global Opinion QA (India Alignment)

```bash
$PYTHON eval_globalopinion.py \
  --num-samples 100 \
  --output results/globalopinion_results.json
```

---

## Files

```
eval_milu.py              # MILU evaluation (MCQ, 4 modes)
eval_normad.py            # NormAd social norm evaluation
eval_bhed.py              # Indian-BhED stereotype score
eval_globalopinion.py     # Global Opinion QA JS-similarity
scripts/start_vllm.sh     # vLLM server startup script
results/
  milu_deepseek_r1_hindi_100.json
  milu_deepseek_r1_english_100.json
  normad_results_fixed.json
  bhed_results.json
  globalopinion_results.json
```

## Dataset Sources

| Dataset | Source |
|---------|--------|
| MILU | `ai4bharat/MILU` (HuggingFace) |
| NormAd | `akhilayerukola/NormAd` (HuggingFace) |
| Indian-BhED | GitHub: `khyatikhandelwal/Indian-LLMs-Bias` |
| Global Opinion QA | `Anthropic/llm_global_opinions` (HuggingFace) |
