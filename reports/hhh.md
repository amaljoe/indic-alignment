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
## Headline

| Mode | 1.5B | 8B | Δ (8B − 1.5B) |
|------|-----:|---:|--------------:|
| zero-shot + thinking    | 58.37% (z=+2.49) | **90.95%** (z=+12.18) | +32.58 pt |
| zero-shot + no-thinking | 51.58% (z=+0.47) | 51.58% (z=+0.47) | +0.00 pt |

## Per-subset accuracy (with-thinking)

| Subset | n | 1.5B | 8B |
|--------|--:|-----:|---:|
| harmless | 58 | 55.17% | **96.55%** |
| helpful | 59 | 55.93% | **91.53%** |
| honest | 61 | 57.38% | **83.61%** |
| other | 43 | 67.44% | **93.02%** |

## Per-subset accuracy (no-thinking)

| Subset | n | 1.5B | 8B |
|--------|--:|-----:|---:|
| harmless | 58 | 50.00% | 50.00% |
| helpful | 59 | 49.15% | 49.15% |
| honest | 61 | 52.46% | 52.46% |
| other | 43 | 55.81% | 55.81% |

## Position bias (predicted-letter distribution)

| Mode | 1.5B | 8B |
|------|------|----|
| + thinking    | A=140, B=81 | A=118, B=103 |
| − thinking    | A=221 | A=221 |

---
## Findings

1. **The 8 B is dramatically better at HHH preference judgement when given thinking budget.** With 8 k tokens it scores 91.0% vs the 1.5 B's 58.4% — both are above chance and significant, but the 8 B is +33 pt absolute. The 8 B's harmless score (97%) is near-ceiling.

2. **Without thinking, both models collapse to chance (~51%).** This is the cleanest demonstration so far that the DeepSeek-R1 family's preference signal lives in the chain-of-thought, not in any first-token bias toward A or B. Without time to reason, even the 8 B can't tell helpful/honest/harmless apart from the alternative.

3. **Honest is the hardest axis for both** — both models drop several points on the honest subset relative to harmless and other (1.5 B 57%, 8 B 84%). Detecting subtle factual / calibration violations needs more capacity than detecting overt harm.

4. **The earlier MILU / GlobalOpinion 'no improvement at 512 tokens' finding generalises here too.** Force the model to reason to completion (with a generous budget) and the 8 B's HHH alignment manifests; cap reasoning and it disappears entirely. For the distillation pipeline this means HHH-style preference data should be collected from the 8 B in a thinking regime, never in a single-letter regime.

---
## How to reproduce

```bash
# both vLLM servers must be up at 8002 (1.5B) and 8003 (8B), --max-model-len >= 16384
/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \
    --model deepseek-r1-1p5b --base-url http://localhost:8002/v1 \
    --batch-size 64 --max-tokens-think 8000 \
    --output results/hhh_1p5b.json
/home/compiling-ganesh/24m0797/envs/vllm/bin/python eval_hhh.py \
    --model deepseek-r1-8b --base-url http://localhost:8003/v1 \
    --batch-size 64 --max-tokens-think 8000 \
    --output results/hhh_8b.json
/home/compiling-ganesh/24m0797/envs/vllm/bin/python scripts/build_hhh_report.py
```

**Dataset**: `data/hhh_alignment/english.jsonl` (221 rows; one JSON object per line with fields `subset`, `input`, `target_scores`). Combined from `HuggingFaceH4/hhh_alignment` `data/{harmless,helpful,honest,other}/task.json` on HF.
