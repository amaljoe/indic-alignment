#!/bin/bash
# Re-run all four benchmarks on the 8B server with thinking enabled and a
# generous max_tokens=8000 budget.
# Requires the 8B vLLM to be running with --max-model-len >= 16384.
# Usage: bash scripts/run_8b_think8k.sh
set -e

NAME=deepseek-r1-8b
PORT=8003
SUB=baseline_8b_think8k
NSAMPLES_MILU=${NSAMPLES_MILU:-100}
NSAMPLES_GO=${NSAMPLES_GO:-100}
SEED=${SEED:-42}
MT=${MT:-8000}                      # generation budget (per sample)

PYTHON=/home/compiling-ganesh/24m0797/envs/vllm/bin/python
ROOT=/home/compiling-ganesh/24m0797/workspace/indic-alignment
OUT=$ROOT/results/$SUB
mkdir -p "$OUT"

BASE_URL=http://localhost:$PORT/v1
cd "$ROOT"

echo "=== MILU Hindi  (think only, mt=$MT) ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_milu.py --model "$NAME" --base-url "$BASE_URL" \
    --language Hindi --num-samples $NSAMPLES_MILU --seed $SEED \
    --batch-size 32 \
    --modes zt ft \
    --max-tokens-think $MT \
    --output "$OUT/milu_hindi.json" 2>&1 | tee -a "$OUT/run.log"

echo "=== MILU English (think only, mt=$MT) ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_milu.py --model "$NAME" --base-url "$BASE_URL" \
    --language English --num-samples $NSAMPLES_MILU --seed $SEED \
    --batch-size 32 \
    --modes zt ft \
    --max-tokens-think $MT \
    --output "$OUT/milu_english.json" 2>&1 | tee -a "$OUT/run.log"

echo "=== NormAd Indic (mt=$MT) ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_normad.py --model "$NAME" --base-url "$BASE_URL" \
    --countries india pakistan bangladesh nepal sri_lanka \
    --seed $SEED --batch-size 32 --max-tokens $MT \
    --output "$OUT/normad.json" 2>&1 | tee -a "$OUT/run.log"

echo "=== Indian-BhED (mt=$MT) ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_bhed.py --model "$NAME" --base-url "$BASE_URL" \
    --batch-size 32 --max-tokens $MT \
    --output "$OUT/bhed.json" 2>&1 | tee -a "$OUT/run.log"

echo "=== Global Opinion (India, mt=$MT) ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_globalopinion.py --model "$NAME" --base-url "$BASE_URL" \
    --num-samples $NSAMPLES_GO --seed $SEED \
    --batch-size 32 --max-tokens $MT \
    --output "$OUT/globalopinion.json" 2>&1 | tee -a "$OUT/run.log"

echo "DONE: $NAME -> $OUT"
