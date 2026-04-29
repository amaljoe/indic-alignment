# Indic Alignment Results

---

## Phase 1 — MILU (Factual Knowledge)

**Method:** Supervised finetuning (SFT) with LoRA on 5,000 MILU examples (2,500 Hindi + 2,500 English).  
**Eval:** 500 examples from MILU test split (250 Hindi + 250 English). No-think mode.

### Overall Accuracy

| Language | Baseline | Post-SFT | Δ |
|----------|----------|----------|---|
| Hindi | 58.0% | - | - |
| English | 65.2% | - | - |
| **Average** | **61.6%** | **-** | **-** |

### Baseline — Per Domain

| Domain | Hindi (correct/total) | Hindi Acc | English (correct/total) | English Acc |
|--------|-----------------------|-----------|-------------------------|-------------|
| Arts & Humanities | 52/78 | 66.7% | 8/20 | 40.0% |
| Business Studies | 3/6 | 50.0% | 8/14 | 57.1% |
| Engineering & Tech | 22/41 | 53.7% | 63/87 | 72.4% |
| Environmental Sciences | 6/11 | 54.5% | 7/7 | 100.0% |
| Health & Medicine | 0/1 | 0.0% | 3/3 | 100.0% |
| Law & Governance | 1/1 | 100.0% | 0/2 | 0.0% |
| Science | 58/105 | 55.2% | 72/113 | 63.7% |
| Social Sciences | 3/7 | 42.9% | 2/4 | 50.0% |

> Chance baseline: 25.0%

---

## Phase 2 — Cultural Reasoning (Distillation)

**Method:** Rejection-sampled distillation from Gemma 3 27B + SFT with LoRA.  
**Eval:** NormAd (5 Indic countries), Indian-BhED (caste + religion bias), GlobalOpinionQA (India subset).  
**Mode:** Think mode (model generates `<think>...</think>` reasoning traces).

### NormAd — Social Norm Acceptability

| | Baseline | Post-Distill | Δ |
|-|----------|-------------|---|
| Accuracy | 69.8% | - | - |
| Macro-F1 | - | - | - |

#### Baseline — Per Country

| Country | Correct/Total | Accuracy |
|---------|--------------|----------|
| Bangladesh | 25/33 | 75.8% |
| India | 20/29 | 69.0% |
| Nepal | 25/37 | 67.6% |
| Pakistan | 24/35 | 68.6% |
| Sri Lanka | 24/35 | 68.6% |

> Majority-class baseline: ~50%

### Indian-BhED — Stereotype Score (↓ better, 50% = random)

| Category | Baseline | Post-Distill | Δ |
|----------|----------|-------------|---|
| Caste | 44.3% | - | - |
| Religion | 50.4% | - | - |
| **Overall** | **47.6%** | **-** | **-** |

> Random baseline: 50.0%

### GlobalOpinionQA — Jensen-Shannon Similarity (↑ better)

| | Baseline | Post-Distill | Δ |
|-|----------|-------------|---|
| JS Similarity (India) | 0.673 | - | - |

> 0 = no overlap, 1 = perfect match with India population distribution

---

## Phase 3 — Safety Alignment (DPO)

**Method:** Direct Preference Optimization (DPO) with LoRA on ~19k HH-RLHF pairs across 7 languages.  
**Eval:** HHH alignment benchmark (221 examples per language). No-think mode.  
**Languages (baseline):** English, Hindi, Malayalam. Tamil, Bengali, Telugu, Marathi added after Phase 3 datagen.

### Overall Accuracy (↑ better, 50% = random)

| Language | Baseline | Post-DPO | Δ |
|----------|----------|----------|---|
| English | 88.2% | - | - |
| Hindi | 84.4% | - | - |
| Malayalam | 80.9% | - | - |
| Tamil | - | - | - |
| Bengali | - | - | - |
| Telugu | - | - | - |
| Marathi | - | - | - |
| **Average** | **84.5%** | **-** | **-** |

### Baseline English — Per Subset

| Subset | Correct/Total | Accuracy |
|--------|--------------|----------|
| Harmless | 56/58 | 96.6% |
| Helpful | 52/59 | 88.1% |
| Honest | 49/61 | 80.3% |
| Other | 38/43 | 88.4% |

### Baseline Hindi — Per Subset

| Subset | Correct/Total | Accuracy |
|--------|--------------|----------|
| Harmless | 57/58 | 98.3% |
| Helpful | 45/56 | 80.4% |
| Honest | 47/61 | 77.0% |
| Other | 35/43 | 81.4% |

### Baseline Malayalam — Per Subset

| Subset | Correct/Total | Accuracy |
|--------|--------------|----------|
| Harmless | 53/58 | 91.4% |
| Helpful | 42/58 | 72.4% |
| Honest | 48/61 | 78.7% |
| Other | 35/43 | 81.4% |

> Chance baseline: 50.0%
