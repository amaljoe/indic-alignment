#!/bin/bash
# Run all four quantitative benchmarks against ONE model at a given vLLM endpoint.
#
# Usage:
#   bash scripts/run_baseline_evals.sh <served_name> <port> <out_subdir>
# e.g.
#   bash scripts/run_baseline_evals.sh deepseek-r1-1p5b 8002 baseline_1p5b
#   bash scripts/run_baseline_evals.sh deepseek-r1-8b   8003 baseline_8b
set -e

NAME=${1:-deepseek-r1-1p5b}
PORT=${2:-8002}
SUB=${3:-baseline_1p5b}
NSAMPLES_MILU=${NSAMPLES_MILU:-100}
NSAMPLES_GO=${NSAMPLES_GO:-100}
SEED=${SEED:-42}
BATCH_MILU=${BATCH_MILU:-128}
BATCH_NORMAD=${BATCH_NORMAD:-64}
BATCH_BHED=${BATCH_BHED:-64}
BATCH_GO=${BATCH_GO:-64}

PYTHON=/home/compiling-ganesh/24m0797/envs/vllm/bin/python
ROOT=/home/compiling-ganesh/24m0797/workspace/indic-alignment
OUT=$ROOT/results/$SUB
mkdir -p "$OUT"

BASE_URL=http://localhost:$PORT/v1

export HF_HOME=${HF_HOME:-/home/compiling-ganesh/24m0797/.cache/huggingface}

cd "$ROOT"

echo "=== MILU Hindi  ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_milu.py --model "$NAME" --base-url "$BASE_URL" \
    --language Hindi --num-samples $NSAMPLES_MILU --seed $SEED \
    --batch-size $BATCH_MILU \
    --output "$OUT/milu_hindi.json" 2>&1 | tee -a "$OUT/run.log"

echo "=== MILU English ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_milu.py --model "$NAME" --base-url "$BASE_URL" \
    --language English --num-samples $NSAMPLES_MILU --seed $SEED \
    --batch-size $BATCH_MILU \
    --output "$OUT/milu_english.json" 2>&1 | tee -a "$OUT/run.log"

echo "=== NormAd Indic ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_normad.py --model "$NAME" --base-url "$BASE_URL" \
    --countries india pakistan bangladesh nepal sri_lanka \
    --seed $SEED --batch-size $BATCH_NORMAD \
    --output "$OUT/normad.json" 2>&1 | tee -a "$OUT/run.log"

echo "=== Indian-BhED ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_bhed.py --model "$NAME" --base-url "$BASE_URL" \
    --batch-size $BATCH_BHED \
    --output "$OUT/bhed.json" 2>&1 | tee -a "$OUT/run.log"

echo "=== Global Opinion (India) ===" | tee -a "$OUT/run.log"
"$PYTHON" eval_globalopinion.py --model "$NAME" --base-url "$BASE_URL" \
    --num-samples $NSAMPLES_GO --seed $SEED \
    --batch-size $BATCH_GO \
    --output "$OUT/globalopinion.json" 2>&1 | tee -a "$OUT/run.log"

echo "DONE: $NAME -> $OUT"
