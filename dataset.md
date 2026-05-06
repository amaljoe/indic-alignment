# Dataset Splits

All splits are deterministic. Phase 2 uses `_train_eval_split(seed=42)` in both
`phase2_train.py` and `phase2_eval.py`; running either script independently produces
the same partition.

---

## Phase 1 — MILU

**Source:** `ai4bharat/MILU` (HuggingFace)  
**Split method:** Official HuggingFace splits — no manual partitioning needed.

| Split | Source HF split | Languages | Samples |
|-------|-----------------|-----------|---------|
| Train (SFT) | `train` + `validation` | Hindi, English | 2,500 + 2,500 = **5,000** |
| Eval | `test` | Hindi, English | 250 + 250 = **500** |
| Internal val (during training) | first 5% of SFT JSONL | Hindi, English | ~250 |

No overlap — train+validation and test are disjoint by dataset design.

---

## Phase 2 — Cultural / Bias / Stereotype

All three Phase 2 datasets have only a `"train"` split (or no HF split at all).
A deterministic 80/20 or 50/50 split is applied in code before any teacher inference
or evaluation, using `_train_eval_split(rows, eval_frac, seed=42)`.

### NormAd

**Source:** `akhilayerukola/NormAd` (HuggingFace, `"train"` split only)  
**Countries:** india, pakistan, bangladesh, nepal, sri_lanka  
**Split method:** 80/20 deterministic shuffle (seed=42)

| Split | Approx samples | Notes |
|-------|---------------|-------|
| Train | ~1,120 | Teacher (Gemma 3 27B) inference + rejection filter → SFT targets |
| Eval | ~280 | Model evaluated directly; no teacher involved |

Actual counts depend on HuggingFace dataset version at download time.

### Indian-BhED

**Source:** GitHub CSVs — `Caste.csv` (106 rows) + `India_Religious.csv` (123 rows)  
**Split method:** 50/50 deterministic shuffle (seed=42) — 50/50 used because 229 total rows is too small for 80/20 to give a meaningful eval set

| Split | Caste | Religion | Total |
|-------|-------|----------|-------|
| Train | ~53 | ~62 | **~115** |
| Eval | ~53 | ~61 | **~114** |

### GlobalOpinionQA

**Source:** `Anthropic/llm_global_opinions` (HuggingFace, `"train"` split only)  
**Filtered to:** India (Current national sample) — ~766 eligible rows  
**Split method:** 80/20 deterministic shuffle (seed=42) on the India-eligible pool

| Split | Samples | Notes |
|-------|---------|-------|
| Train | ~613 | All rows in the 80% pool; teacher inference + rejection filter applied |
| Eval | ~153 | All rows in the 20% pool; evaluated directly |

---

## Phase 3 — Safety (HHH Alignment)

**No manual splitting needed** — training and evaluation use entirely different datasets.

### Training — HH-RLHF

**Source:** Anthropic HH-RLHF preference pairs (local JSONL files in `data/hh_rlhf/`)

| File | Language | DPO pairs |
|------|----------|-----------|
| `hh_rlhf_5k_en.jsonl` | English | 5,000 |
| `hh_rlhf_5k_hindi.jsonl` | Hindi | 5,000 |
| `hh_rlhf_5k_malayalam.jsonl` | Malayalam | 5,000 |
| `hh_rlhf_2k_tamil.jsonl` | Tamil | ~2,000 |
| `hh_rlhf_2k_bengali.jsonl` | Bengali | ~2,000 |
| `hh_rlhf_2k_telugu.jsonl` | Telugu | ~2,000 |
| `hh_rlhf_2k_marathi.jsonl` | Marathi | ~2,000 |
| **Total** | | **~23,000** |

### Evaluation — HHH Alignment

**Source:** `HuggingFaceH4/hhh_alignment` (local JSONL files in `data/hhh_alignment/`)

| File | Language | Examples |
|------|----------|----------|
| `english.jsonl` | English | 221 |
| `hindi_gemma3_27b.jsonl` | Hindi | 221 |
| `malayalam_gemma3_27b.jsonl` | Malayalam | 221 |
| `tamil_gemma3_27b.jsonl` | Tamil | 221 |
| `bengali_gemma3_27b.jsonl` | Bengali | 221 |
| `telugu_gemma3_27b.jsonl` | Telugu | 221 |
| `marathi_gemma3_27b.jsonl` | Marathi | 221 |
| **Total** | | **1,547** |

HH-RLHF (preference pairs) and HHH (binary forced-choice) are different datasets with no overlap.

---

## Summary

| Phase | Dataset | Train samples | Eval samples | Split method | Clean? |
|-------|---------|---------------|--------------|--------------|--------|
| 1 | MILU Hindi | 2,500 | 250 | Official `train`+`val` / `test` | Yes |
| 1 | MILU English | 2,500 | 250 | Official `train`+`val` / `test` | Yes |
| 2 | NormAd | ~1,120 | ~280 | 80/20 code split (seed=42) | Yes (fixed) |
| 2 | BhED | ~115 | ~114 | 50/50 code split (seed=42) | Yes (fixed) |
| 2 | GlobalOpinionQA | ~613 | ~153 | 80/20 pool split (seed=42) | Yes (fixed) |
| 3 | HH-RLHF / HHH | ~23,000 | 1,547 | Different datasets entirely | Yes |
