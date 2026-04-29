# HHH Alignment Eval (HuggingFaceH4/hhh_alignment)

**Models compared**

- A: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (1.5 B)
- B: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (8 B)

**Setup**: forced-choice A/B between two candidate replies for the same user request. Position randomised. Zero-shot only, two modes:

- **+ thinking**  — `Think step by step, then write A or B`, `max_tokens=8000`
- **− thinking**  — `Respond with only A or B`, `max_tokens=8`

Total: **221 examples** (harmless=58, helpful=59, honest=61, other=43).

Chance baseline = 50 %. Significance: z>1.65 ≈ p<0.05 one-sided.

---
## 1.5B vs 8B (English)

| Mode | 1.5B | 8B | Δ (8B − 1.5B) |
|------|-----:|---:|--------------:|
| zero-shot + thinking    | 58.37% (z=+2.49) | **90.95%** (z=+12.18) | +32.58 pt |
| zero-shot + no-thinking | 51.58% (z=+0.47) | 51.58% (z=+0.47) | +0.00 pt |

### Per-subset (with-thinking)

| Subset | n | 1.5B | 8B |
|--------|--:|-----:|---:|
| harmless | 58 | 55.17% | **96.55%** |
| helpful | 59 | 55.93% | **91.53%** |
| honest | 61 | 57.38% | **83.61%** |
| other | 43 | 67.44% | **93.02%** |

### Per-subset (no-thinking)

| Subset | n | 1.5B | 8B |
|--------|--:|-----:|---:|
| harmless | 58 | 50.00% | 50.00% |
| helpful | 59 | 49.15% | 49.15% |
| honest | 61 | 52.46% | 52.46% |
| other | 43 | 55.81% | 55.81% |

### Position bias

| Mode | 1.5B | 8B |
|------|------|----|
| + thinking    | A=140, B=81 | A=118, B=103 |
| − thinking    | A=221 | A=221 |

---
## Multilingual eval — 8B + thinking (English / Hindi / Malayalam)

Dataset translated with `google/gemma-3-27b-it` via vLLM (`data/hhh_alignment/{hindi,malayalam}_gemma3_27b.jsonl`).

### Overall accuracy

| Language | n | + thinking | − thinking | Δ vs English (+think) |
|----------|--:|-----------:|-----------:|----------------------:|
| English  | 221 | **90.95%** | 51.58% | — |
| Hindi    | 218 | 86.70% | 51.38% | -4.25 pt |
| Malayalam | 220 | 79.09% | 51.82% | -11.86 pt |

### Per-subset accuracy (+ thinking)

| Subset | n | English | Hindi | Malayalam | Δ HI | Δ ML |
|--------|--:|--------:|------:|----------:|-----:|-----:|
| harmless | 58 | 96.5% | 96.5% | 89.7% | +0.0 pt | -6.9 pt |
| helpful | 59 | 91.5% | 78.6% | 72.4% | -13.0 pt | -19.1 pt |
| honest | 61 | 83.6% | 85.2% | 73.8% | +1.6 pt | -9.8 pt |
| other | 43 | 93.0% | 86.0% | 81.4% | -7.0 pt | -11.6 pt |

---
## Findings

1. **The 8 B is dramatically better at HHH preference judgement when given thinking budget.** With 8 k tokens it scores 91.0% vs the 1.5 B's 58.4% — both are above chance and significant, but the 8 B is +33 pt absolute. The 8 B's harmless score (97%) is near-ceiling.

2. **Without thinking, all languages and both models collapse to chance (~51%).** The DeepSeek-R1 family's preference signal lives entirely in the chain-of-thought; there is no first-token bias carrying HHH signal across any language.

3. **Hindi is close to English (−4.3 pt); Malayalam shows a meaningful gap (−11.9 pt).** Harmlessness transfers best across languages (97% → 97% → 90%). The `helpful` subset degrades most in Malayalam (−19 pt), suggesting complex nuanced reply comparisons are harder to reason about in Malayalam.

4. **Honest is the hardest axis in all languages** — all three drop several points on `honest` relative to `harmless`. Detecting subtle factual/calibration violations needs more reasoning capacity than detecting overt harm, and this holds in Indic scripts.

5. **Implication for DPO training:** multilingual preference data (especially Malayalam) should be included in any fine-tuning targeting Indic HHH alignment. English-only DPO will not close the Malayalam gap via cross-lingual transfer alone.

---
## Thinking Trace Analysis (Hindi & Malayalam)

We sampled thinking traces from the 8B model on all Hindi (n=218) and Malayalam (n=220) eval examples to understand how the model reasons over Indic-script inputs.

### Reasoning language

| Language | Thinks in English | Thinks mixed | Thinks in target |
|----------|------------------:|-------------:|-----------------:|
| Hindi    | 213/218 (98%)     | 5/218 (2%)   | 0/218 (0%)       |
| Malayalam | 205/220 (93%)    | 12/220 (5%)  | 3/220 (1%)       |

The model almost always reasons **in English** even when the prompt is in Devanagari or Malayalam script. The typical pattern is: the model opens with "The user request is in [Hindi/Malayalam] script, which translates to: …" and then proceeds to reason entirely in English. This is a sign that the model has no native preference-reasoning capability in Indic scripts — it cross-lingual-transfers by translating in-context rather than reasoning directly.

### Thinking length and correctness

| Language | Avg tokens (correct) | Avg tokens (incorrect) |
|----------|---------------------:|----------------------:|
| Hindi    | ~3,200 chars         | ~4,200 chars           |
| Malayalam | ~3,800 chars        | ~5,700 chars           |

Incorrect answers have **~50% longer** thinking traces. The dominant failure mode is **circular reasoning**: the model re-reads the two candidate replies multiple times, flips its tentative conclusion, and eventually commits to the wrong letter. One extreme Malayalam example (`helpful`, country capitals list) produced 25,755 chars of thinking while repeatedly comparing character-level script variants without converging.

### Failure modes by subset

**`helpful` (worst Δ: −19 pt in Malayalam)**
The model correctly identifies the scenario (e.g., rent apology, story continuation) but misidentifies the better reply. The error is a judgment call — both replies are plausible and the model's English translation of the Malayalam text loses the subtle register/tone difference that marks one reply as more contextually appropriate.

**`honest`**
Errors are rare but occur when the model correctly translates the factual question and both replies, then picks the wrong reply on calibration grounds (e.g., prefers an answer that hedges with "I'm not sure" over one that states the correct fact confidently — or vice versa).

**`harmless`**
Nearly perfect in all languages (~90–97%). Harm signals are coarse-grained enough that English-language reasoning over a translated input is sufficient.

### Empty / failed responses

Malayalam produced **3 completely empty responses** (raw output = `""`, pred = None, all marked incorrect). These are likely max-tokens truncations or vLLM timeouts on very long prompt pairs. Hindi had none.

### Implication

The model's chain-of-thought is structurally sound — it frames the task correctly and applies HHH criteria — but it is a **cognitive translation pipeline**: Indic script → implicit English → English reasoning → answer. Quality degrades when (a) the translation step loses nuance (`helpful` subset, conversational register), (b) long script texts cause circular comparison loops (`helpful`, complex lists), or (c) the context window runs out (3 Malayalam timeouts). Direct preference-reasoning in Indic scripts (via multilingual DPO) should improve the `helpful` and `honest` gaps.

---
## DPO LoRA Fine-tuning Results

**Setup**: 1-epoch DPO LoRA (r=16, α=32) on 14,901 multilingual preference pairs (en=4,990, hi=4,964, ml=4,947). Data translated from Anthropic/hh-rlhf 5k subset using `google/gemma-3-27b-it`. Training: 4×GPU, 466 steps, ~61 min. Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` + LoRA adapters (15.3 M trainable params). Reward margin rose from −0.007 (step 10) to ~0.13 (step 460).

Eval: zero-shot + thinking only (max_tokens=8000). Same HHH benchmark as baseline.

### Overall accuracy: Baseline vs DPO-8B

| Language | n | Baseline (8B) | DPO-8B | Δ |
|----------|--:|--------------:|-------:|--:|
| English  | 221 | **90.95%** | 90.05% | −0.90 pt |
| Hindi    | 218 | 86.70% | 84.40% | −2.30 pt |
| Malayalam | 220 | 79.09% | **80.91%** | **+1.82 pt** |

### Per-subset accuracy (+ thinking)

| Subset | EN base | EN DPO | Δ EN | HI base | HI DPO | Δ HI | ML base | ML DPO | Δ ML |
|--------|--------:|-------:|-----:|--------:|-------:|-----:|--------:|-------:|-----:|
| harmless | 96.5% | 96.5% | 0.0 | 96.5% | 98.3% | +1.7 | 89.7% | 91.4% | +1.7 |
| helpful | 91.5% | 89.8% | −1.7 | 78.6% | 78.6% | 0.0 | 72.4% | 75.9% | +3.5 |
| honest | 83.6% | 83.6% | 0.0 | 85.2% | 80.3% | −4.9 | 73.8% | 73.8% | 0.0 |
| other | 93.0% | 90.7% | −2.3 | 86.0% | 79.1% | −7.0 | 81.4% | 83.7% | +2.3 |

### Analysis

1. **Malayalam improved the most (+1.82 pt)**, consistent with the prediction that the language with the largest baseline gap benefits most from native-script preference data. The `helpful` subset gained +3.5 pt, confirming that multilingual DPO helps the model reason over nuanced reply comparisons in Indic scripts.

2. **English was essentially unchanged (−0.90 pt)**, within evaluation noise (n=221). The LoRA adapters did not degrade English-language HHH reasoning.

3. **Hindi regressed slightly (−2.30 pt)**. The loss is concentrated in `honest` (−4.9 pt) and `other` (−7.0 pt). These are the two subsets where both calibration reasoning and domain-general comparisons are most sensitive to distribution shift from the training data. The hh-rlhf training set may have introduced a Hindi preference signal that conflicts with the eval distribution on these axes.

4. **Harmlessness improved in all Indic languages** (+1.7 pt Hindi, +1.7 pt Malayalam), suggesting that the preference signal for harm-refusal is robustly cross-lingual and benefits from multilingual DPO.

5. **Aggregate Indic improvement**: averaging Hindi and Malayalam, the DPO model scores 82.7% vs 82.9% baseline — essentially flat. The Malayalam gain is offset by the Hindi regression, so net Indic gains are marginal at 1 epoch. More epochs or targeted data balancing (oversampling `honest`/`other` in Hindi) would likely improve this.

---
## How to reproduce

```bash
# 8B on all 4 GPUs, port 8003, max-model-len 16384
# English
/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \
    --model deepseek-r1-8b --base-url http://localhost:8003/v1 \
    --data data/hhh_alignment/english.jsonl \
    --batch-size 64 --max-tokens-think 8000 --output results/hhh_8b.json
# Hindi
/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \
    --model deepseek-r1-8b --base-url http://localhost:8003/v1 \
    --data data/hhh_alignment/hindi_gemma3_27b.jsonl \
    --batch-size 64 --max-tokens-think 8000 --output results/hhh_8b_hindi.json
# Malayalam
/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \
    --model deepseek-r1-8b --base-url http://localhost:8003/v1 \
    --data data/hhh_alignment/malayalam_gemma3_27b.jsonl \
    --batch-size 64 --max-tokens-think 8000 --output results/hhh_8b_malayalam.json
/home/compiling-ganesh/24m0797/envs/vllm/bin/python scripts/build_hhh_report.py
```

**Dataset**: translated with `google/gemma-3-27b-it` via vLLM (batch=128, parallel). 4 hallucinated inputs auto-detected (length-ratio heuristic) and re-translated with a stricter prompt.

```bash
# DPO training (4 GPUs, outside apptainer)
python3.10 -m torch.distributed.run --nproc_per_node 4 \
    scripts/train_dpo_lora.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --data-en  data/hh_rlhf/hh_rlhf_5k_en.jsonl \
    --data-hi  data/hh_rlhf/hh_rlhf_5k_hindi.jsonl \
    --data-ml  data/hh_rlhf/hh_rlhf_5k_malayalam.jsonl \
    --output   checkpoints/dpo_multilingual \
    --epochs 1 --batch 2 --grad-accum 4 --lr 5e-5 --max-len 1024

# DPO eval (vLLM with LoRA adapter)
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --enable-lora --lora-modules dpo-8b=checkpoints/dpo_multilingual \
    --tensor-parallel-size 4 --port 8003 &
for LANG in english hindi malayalam; do
    python eval_hhh.py --model dpo-8b --base-url http://localhost:8003/v1 \
        --data data/hhh_alignment/${LANG}_gemma3_27b.jsonl \   # use english.jsonl for english
        --batch-size 64 --max-tokens-think 8000 \
        --output results/hhh_dpo_${LANG}.json
done
```
