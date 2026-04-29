# Methodology

## Overview

We evaluate and finetune **DeepSeek-R1-0528-Qwen3-8B** on three orthogonal axes of Indic alignment using a sequential three-phase pipeline. Each phase targets a distinct alignment problem with a tailored finetuning method.

---

## Model

**DeepSeek-R1-0528-Qwen3-8B** — a 8B-parameter reasoning model with native `<think>...</think>` chain-of-thought support. We exploit its thinking capability selectively:

| Phase | Thinking | Rationale |
|-------|----------|-----------|
| 1 · MILU | Off (no-think) | MCQ recall benefits from direct answer; thinking adds noise |
| 2 · Cultural | On (think) | Cultural reasoning requires multi-step inference |
| 3 · Safety | Off (no-think) | Preference selection is fast; thinking not needed |

---

## Training Setup

All phases use **LoRA** (Low-Rank Adaptation):

| Hyperparameter | Value |
|----------------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |

LoRA adapters are loaded hot into the running vLLM server (`POST /v1/load_lora_adapter`) — no model restart required between phases.

Infrastructure: Apptainer container, `~/envs/vllm`, 2× training GPUs (torchrun), 2× eval GPUs (vLLM).

---

## Phase 1 — Factual Knowledge (SFT)

**Benchmark:** MILU (Multitask Indian Language Understanding)  
**Dataset:** ai4bharat/MILU — 85k questions across 8 domains, 11 Indic languages  
**Train split:** 2,500 Hindi + 2,500 English from the official train split  
**Eval split:** 250 Hindi + 250 English from the official test split  
**Method:** Supervised finetuning (SFT) with LoRA

Training data format — each example is a direct question→letter pair with a no-think system prompt:
```
[system]: Respond with ONLY the single letter A, B, C, or D.
[user]:   Question: Which Indian state has the longest coastline?
          A. Tamil Nadu  B. Gujarat  C. Andhra Pradesh  D. Maharashtra
          Answer:
[asst]:   B
```

Gold answer letters (A/B/C/D) are used directly as training targets. No teacher model is needed. Sequence length is kept at 512 tokens since questions are short.

**Key design choice:** Matching train format exactly to eval format (same system prompt, same no-think mode) ensures the model adapts correctly without mode confusion.

---

## Phase 2 — Cultural Reasoning (Distillation)

**Benchmarks:** NormAd · Indian-BhED · GlobalOpinionQA  
**Teacher:** Gemma 3 27B (google/gemma-3-27b-it)  
**Method:** Rejection-sampled chain-of-thought distillation + SFT with LoRA

### Distillation pipeline

```
01 · SAMPLE   → train split of each dataset
      ↓
02 · PROMPT   → Gemma 3 27B with full cultural context + <think> instruction
                 temp=0.3, max_tokens=4096
      ↓
03 · FILTER   → keep only samples where Gemma's answer matches ground truth
      ↓
04 · DISTILL  → SFT DeepSeek on (student_input → Gemma_think_trace + answer)
```

### Context distillation trick

The teacher (Gemma) sees **full context** (e.g., cultural Background for NormAd), while the student training input deliberately **omits** that context. The student must learn to reason from first principles what the teacher learned from explicit hints. This forces genuine cultural knowledge acquisition.

| Dataset | Teacher prompt includes | Student prompt includes |
|---------|------------------------|-------------------------|
| NormAd | Country + Cultural Background + Story | Story only |
| BhED | Full sentence + fairness framing | Sentence only |
| GlobalOpinion | India-specific framing | Question + options only |

Training data format — each record contains Gemma's complete reasoning trace as the assistant turn:
```
[system]: You are a cultural etiquette expert. Think carefully, then answer.
[user]:   Story: In an Indian household, a guest is offered tea...
[asst]:   <think>
          In Indian hospitality, an initial polite refusal is conventional...
          </think>
          yes
```

**Sequence length:** 4096 tokens (Gemma traces can be 500–2000 tokens). Gradient checkpointing is mandatory at this length with an 8B model.

---

## Phase 3 — Safety Alignment (DPO)

**Benchmark:** HHH Alignment (HuggingFaceH4/hhh_alignment) — 7 languages  
**Training data:** Anthropic HH-RLHF translated to 7 Indic languages  
**Method:** Direct Preference Optimization (DPO) with LoRA

### Dataset

Existing translations (en, hi, ml) come from the repository's `data/hh_rlhf/` directory. Four additional languages (Tamil, Bengali, Telugu, Marathi) are translated using Gemma 3 27B:

- **Translation batching:** 128 concurrent API calls to Gemma (one sample per call). vLLM queues excess requests internally.
- **Format preservation:** Conversation markers (`\n\nHuman:`, `\n\nAssistant:`) are preserved verbatim.
- **HHH eval translation:** The 221-example HHH eval set is also translated to the 4 new languages for post-DPO multilingual evaluation.

### DPO training

Preference pairs follow the standard hh-rlhf format:
```json
{
  "prompt":   "\n\nHuman: <user request>\n\nAssistant:",
  "chosen":   "<helpful, harmless, honest response>",
  "rejected": "<less aligned response>"
}
```

DPO hyperparameters: β=0.1, max_seq_len=2048, 3 epochs, lr=5e-5.

---

## Evaluation Protocol

All evaluations use vLLM served via the OpenAI-compatible API with 128 parallel requests (ThreadPoolExecutor + as_completed). Temperature is fixed at 0.0.

| Phase | Metric | Baseline | Mode |
|-------|--------|----------|------|
| MILU | Accuracy (4-way MCQ) | chance=25% | no-think |
| NormAd | Accuracy + Macro-F1 | majority-class | think |
| BhED | Stereotype Score (↓ = better) | 50%=random | think |
| GlobalOpinion | Jensen-Shannon similarity (↑) | 0=no match | think |
| HHH | Accuracy (binary choice) | chance=50% | no-think |

**Output validation** runs on every inference call: overflow detection (missing `</think>` in think mode, `finish_reason=length`), gibberish detection (low alpha ratio, repeated character runs), and empty-response detection. An alert fires if overflow exceeds 20% in any batch.
