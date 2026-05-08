# Indic Alignment of DeepSeek-R1-0528-Qwen3-8B

Post-training pipeline that aligns an 8B reasoning model across three orthogonal axes of Indic alignment: factual knowledge, cultural understanding, and multilingual safety.

**Model on HuggingFace:** [amaljoe88/deepseek-r1-8b-indic-aligned](https://huggingface.co/amaljoe88/deepseek-r1-8b-indic-aligned)

---

## Results

| Phase         | Benchmark                               | Before | After | Δ       |
| ------------- | --------------------------------------- | ------ | ----- | ------- |
| 1 — Knowledge | MILU Hindi (accuracy)                   | 47.6%  | 58.0% | +10.4pp |
| 2 — Culture   | BhED Stereotype Score (lower is better) | 44.1%  | 27.9% | −16.2pp |
| 2 — Culture   | GlobalOpinion JS-sim                    | 0.668  | 0.715 | +0.046  |
| 3 — Safety    | HHH avg across 7 languages              | 62.4%  | 79.0% | +16.6pp |

---

## Repository Layout

```
indic-alignment/
├── src/                  # All training, evaluation, and utility scripts
│   ├── phase1_train.py       # Phase 1: SFT on MILU (Hindi factual MCQ)
│   ├── phase1_eval.py        # Phase 1: MILU evaluation
│   ├── phase2_train.py       # Phase 2: Cultural distillation SFT
│   ├── phase2_eval.py        # Phase 2: NormAd / BhED / GlobalOpinion eval
│   ├── phase3_train.py       # Phase 3: DPO on HH-RLHF (7 languages)
│   ├── phase3_eval.py        # Phase 3: HHH alignment eval (7 languages)
│   ├── pipeline_state.py     # Shared pipeline state helpers
│   ├── pipeline_status.py    # Pipeline progress display
│   ├── filter_milu_india.py  # Filter/prepare MILU dataset
│   ├── generate_hhh_multilingual.py  # Translate HHH eval + HH-RLHF to Indic languages
│   ├── push_to_hf.py         # Push LoRA adapter to HuggingFace Hub
│   ├── serve.sh              # Start vLLM server for the student model
│   ├── serve_gemma.sh        # Start vLLM server for Qwen3-8B baseline
│   ├── serve_teacher.sh      # Start vLLM server for Gemma 3 27B teacher
│   ├── run_step.sh           # Run a single pipeline step via torchrun
│   └── run_gemma_eval.sh     # Run Phase 1+2+3 evals for Qwen3-8B baseline
├── data/                     # Training data (gitignored large files)
├── checkpoints/              # LoRA checkpoints (gitignored)
├── results/                  # Eval result JSONs (gitignored)
```

---

## Pipeline Overview

Three sequential phases, each building on the previous checkpoint:

```
Base model (DeepSeek-R1-0528-Qwen3-8B)

Phase 1 — SFT on MILU (Hindi factual MCQ)
Method: Supervised fine-tuning, no-think mode
Data:   2.5k Hindi + 2.5k English MILU train examples

Phase 2 — Cultural distillation SFT
Method: Rejection-sampled CoT distillation from Gemma 3 27B, think mode
Data:   NormAd (Indic) + Indian-BhED + GlobalOpinion India

Phase 3 — DPO on HH-RLHF (7 languages)
Method: Direct Preference Optimization, no-think mode
Data:   ~19k preference pairs (en, hi, ml, ta, bn, te, mr)
```

All phases use **LoRA** (r=16, α=32) loaded hot into vLLM — no model restart between phases.

---

## Running the Pipeline

**Start servers:**

```bash
bash src/serve.sh          # vLLM student model (eval GPUs)
bash src/serve_teacher.sh  # Gemma 3 27B teacher (Phase 2 distillation)
```

**Run a phase:**

```bash
bash src/run_step.sh phase1_train
bash src/run_step.sh phase1_eval
bash src/run_step.sh phase2_train
bash src/run_step.sh phase2_eval
bash src/run_step.sh phase3_train
bash src/run_step.sh phase3_eval
```

**Push to HuggingFace:**

```bash
python src/push_to_hf.py
```

---

## Evaluation Benchmarks

| Phase | Benchmark                                     | Metric                             | Baseline       |
| ----- | --------------------------------------------- | ---------------------------------- | -------------- |
| 1     | MILU Hindi (8 domains, 4-way MCQ)             | Accuracy                           | 25% chance     |
| 2     | NormAd — social acceptability                 | Accuracy                           | majority-class |
| 2     | Indian-BhED — caste/religion stereotypes      | Stereotype Score (lower is better) | 50% random     |
| 2     | GlobalOpinionQA — Indian opinion distribution | Jensen-Shannon sim                 | 0 = no match   |
| 3     | HHH Alignment — 7 languages                   | Accuracy (binary)                  | 50% chance     |
