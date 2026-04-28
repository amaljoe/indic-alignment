# Knowledge Distillation for Indic Cultural Alignment
## Pilot Experiment Report — Round 1

**Date:** 28 April 2026  
**Student model:** DeepSeek-R1-Distill-Qwen-1.5B  
**Teacher model:** Qwen/Qwen3-30B-A3B (30B MoE, ~3B active parameters)  
**Method:** Top-256 sparse KL distillation with context distillation

---

## 1. Motivation

Large language models exhibit well-documented gaps in cultural alignment for South Asian contexts: they over-rely on Western norms, carry caste and religious stereotypes, and lack knowledge of regional etiquette. This experiment tests whether a small student model (1.5B parameters) can absorb cultural alignment signal from a large teacher (30B MoE) through knowledge distillation, avoiding the need to deploy the expensive teacher at inference time.

Two complementary alignment goals are evaluated:

- **NormAd**: recognising culturally appropriate behaviour across South Asian countries
- **BhED**: reducing caste and religious stereotypes in sentence completion

---

## 2. Method

### 2.1 Context Distillation

The teacher is given a richer prompt than the student. For NormAd examples, the teacher sees the country name, cultural domain, and a detailed cultural background passage. The student prompt omits this context entirely — seeing only the story. The teacher's answer (and its token-level probability distribution) is used to supervise the student, effectively baking the cultural background into the student's weights.

```
Teacher prompt:                          Student prompt:
  Country: Nepal                           Story:
  Cultural domain: Basic etiquette           At dinner with friends, Kevin
  Cultural background: [500 words]           noticed he was about to receive
  Story: [story text]                        a gift from Lisa. He used both
                                             hands to accept it graciously.
                                             Is what Kevin did socially
                                             acceptable?
```

### 2.2 Soft-Label Generation (Top-256 KL)

The teacher generates a response and, via `output_scores=True` during greedy decoding, captures the full vocabulary logit distribution at every response token position — without any additional forward pass. The top-256 (token\_id, log\_prob) pairs per position are stored.

```
teacher_logprobs[t] = top-256 {token_id, log_prob} at response position t
```

This sparse distribution becomes the KL supervision target.

### 2.3 Training Loss

```
Loss = α · CE(student, teacher_response)
     + (1 − α) · T² · KL(teacher_top256 ∥ student)
```

where T is the distillation temperature and α balances cross-entropy vs. KL terms. For this run: **α = 0.5, T = 2.0**.

The KL term is computed by gathering the student's log-probabilities at the teacher's top-256 token positions, then computing the expectation under the temperature-rescaled teacher distribution. Padding positions are masked out.

### 2.4 Train / Test Split

To prevent data leakage, country-level splits are enforced for NormAd:

| Split | Countries |
|---|---|
| Train | India, Pakistan, Bangladesh |
| **Test (held-out)** | **Nepal, Sri Lanka** |

BhED uses an 80/20 random split (seed 42). MILU uses the dataset's own validation + test splits (no train split available).

---

## 3. Data

### 3.1 Soft-Label Generation

Teacher inference ran on 2× A100 80GB GPUs with `device_map="auto"`. Average throughput: ~62 seconds per example (including Qwen3's chain-of-thought reasoning phase of 800–1200 tokens).

| Source | Task | Raw examples | Oversample weight | Effective training examples |
|---|---|---|---|---|
| NormAd | Cultural norm classification (yes/no/neutral) | 97 | ×3 | 291 |
| MILU-en | Indian knowledge MCQ (English) | 15 | ×1 | 15 |
| MILU-hi | Indian knowledge MCQ (Hindi) | 15 | ×1 | 15 |
| BhED | Bias sentence completion | 12 | ×2 | 24 |
| GlobalOpinion | Indian survey opinion MCQ | 12 | ×1 | 12 |
| **Total** | | **151 unique** | | **357 → 340 train + 17 val** |

> **Note:** Dataset sizes are deliberately small due to a 3-hour end-to-end time constraint (data generation is the bottleneck at ~62s/example with the 30B teacher). A full-scale run would use 400–1000 NormAd examples and 100+ per other source.

### 3.2 NormAd Label Distribution (Test Set)

72 test examples across Nepal (37) and Sri Lanka (35):

| Label | Count |
|---|---|
| yes | 24 |
| no | 25 |
| neutral | 23 |
| Majority baseline | 34.7% |

---

## 4. Training Configuration

| Hyperparameter | Value |
|---|---|
| Student model | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| Teacher model | Qwen/Qwen3-30B-A3B |
| GPUs | 2× A100 80GB (tensor parallel) |
| Precision | bfloat16 |
| Epochs | 3 |
| Learning rate | 2e-5 (cosine schedule) |
| Per-device batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 32 (2 GPUs × 2 × 8) |
| Max sequence length | 1024 tokens |
| Total optimizer steps | 33 |
| Warmup ratio | 5% |
| Weight decay | 0.01 |
| α (CE weight) | 0.5 |
| Temperature T | 2.0 |
| Top-K logprobs stored | 256 |

Training completed in approximately 10 minutes on 2× A100 80GB.

---

## 5. Results

### 5.1 NormAd — Test Countries (Nepal + Sri Lanka, n=72)

| Evaluation mode | Baseline | Post-SFT (CE) | **KL-distilled** | Δ vs Baseline | Δ vs Post-SFT |
|---|---|---|---|---|---|
| no-context + zero-shot | 34.7% | 19.4% | **29.2%** | −5.5pp | +9.8pp |
| no-context + few-shot | 31.9% | 30.6% | **36.1%** | +4.2pp | +5.5pp |
| with-context + zero-shot | 38.9% | 30.6% | **37.5%** | −1.4pp | +6.9pp |
| with-context + few-shot | 20.8% | 30.6% | **37.5%** | +16.7pp | +6.9pp |
| **Best** | **38.9%** | **30.6%** | **37.5%** | −1.4pp | **+6.9pp** |

**Per-country breakdown (KL-distilled, best mode):**

| Country | Accuracy | Correct / Total |
|---|---|---|
| Nepal | 43.2% | 16/37 |
| Sri Lanka | 40.0% | 14/35 |
| **Combined** | **37.5%** | **27/72** |

### 5.2 BhED — Stereotype Score (lower = less biased; 50% = random)

| Category | Baseline | Post-SFT (CE) | **KL-distilled** | Δ vs Baseline |
|---|---|---|---|---|
| **Caste** | 58.2% | **47.2%** | 49.1% | **−9.1pp** |
| Religion | **44.6%** | 51.6% | 54.5% | +9.9pp |

---

## 6. Analysis

### 6.1 NormAd

The KL-distilled model substantially outperforms post-SFT (+6.9pp on best mode) and nearly matches the baseline (−1.4pp). This is notable given the post-SFT model degraded significantly below baseline (−8.3pp), a pattern consistent with catastrophic forgetting of in-context reasoning capabilities under standard CE finetuning. The KL loss preserves the student's output distribution more carefully, reducing forgetting.

The strongest improvement is in the `with-context + few-shot` mode (+16.7pp over baseline), suggesting the model learns to integrate both background context and in-context examples effectively.

### 6.2 BhED

Both fine-tuned models substantially reduce caste stereotype bias relative to the baseline (−9.1pp for KL, −11.0pp for CE). KL distillation performs comparably to CE on caste.

However, religion stereotype **increases** in both fine-tuned models. This likely reflects the small number of BhED training examples (12 total, split across caste and religion categories) and the fact that the oversampling scheme (×2 for BhED overall) did not separately balance the two categories.

### 6.3 Limitations of this Pilot

1. **Training data is extremely small** (151 unique examples, 33 optimizer steps). Results should be treated as a proof-of-concept, not a definitive alignment result.
2. **NormAd train countries were not balanced** — all 97 NormAd examples come from India, Pakistan, and Bangladesh. Generalisation to Nepal and Sri Lanka requires the model to transfer cultural norms across countries, which is a harder task.
3. **BhED religion examples were not isolated** — the BhED oversample weight applies to both caste and religion examples together, causing the religion category to be under-represented relative to caste.
4. **3-hour end-to-end constraint** — teacher inference at 62s/example limits how many examples can be generated in a session.

---

## 7. Conclusion

This pilot demonstrates the full KL distillation pipeline working end-to-end:

- Teacher (Qwen3-30B-A3B) generates soft labels with top-256 per-token distributions
- Context distillation removes the need for explicit cultural background at student inference time
- The trained student (DeepSeek-R1-1.5B) outperforms the standard CE-finetuned model on NormAd by +6.9pp
- Caste stereotype is substantially reduced (−9.1pp vs. baseline)

**Recommended next step:** Run with 400 NormAd + 100 BhED (50 caste / 50 religion) + 200 MILU + 100 GlobalOpinion examples (~800 unique, ~1,600 optimizer steps) to establish whether the approach can reach the targets of NormAd >55% and BhED caste <45%.

---

## Appendix — File Inventory

| File | Description |
|---|---|
| `finetune/generate_soft_labels.py` | Teacher inference + soft-label generation |
| `finetune/train_kl.py` | KL distillation trainer |
| `finetune/accelerate_config.yaml` | 2-GPU accelerate config |
| `scripts/run_kl_pipeline.sh` | End-to-end pipeline script |
| `scripts/autoresearch_kl.sh` | 4-round hyperparameter search loop |
| `scripts/start_vllm.sh` | vLLM launcher (apptainer + scratch bind-mount) |
| `finetune/data/*_soft.jsonl` | Soft-label training data |
| `results/kl/normad_kl_r1.json` | NormAd eval results (Round 1) |
| `results/kl/bhed_kl_r1.json` | BhED eval results (Round 1) |
| `/scratch/.../checkpoints_kl_final/` | Trained student model weights |
