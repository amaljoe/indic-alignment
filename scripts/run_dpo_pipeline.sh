#!/bin/bash
# Full DPO pipeline: translate → train → eval (thinking-only)
# Run this after translation is done and Gemma 27B is stopped.
#
# Usage:
#   bash scripts/run_dpo_pipeline.sh
#
# Assumes:
#   - data/hh_rlhf/hh_rlhf_5k_{en,hindi,malayalam}.jsonl exist
#   - 4 GPUs available (uses all 4 for 8B, tp=4)
#   - apptainer container with vllm env at ~/envs/vllm
set -e

PYTHON=/home/compiling-ganesh/24m0797/envs/vllm/bin/python
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
SERVED_NAME=deepseek-r1-8b
PORT=8003
TP=4
GPUS=0,1,2,3
LOG_DIR=$ROOT/results

mkdir -p "$LOG_DIR"

echo "=== DPO Pipeline: $(date) ==="

# ── Step 1: DPO Training ───────────────────────────────────────────────────────
echo ""
echo "[1/3] Starting DPO LoRA training..."
torchrun --nproc_per_node $TP \
    "$ROOT/scripts/train_dpo_lora.py" \
    --model "$MODEL" \
    --data-en  "$ROOT/data/hh_rlhf/hh_rlhf_5k_en.jsonl" \
    --data-hi  "$ROOT/data/hh_rlhf/hh_rlhf_5k_hindi.jsonl" \
    --data-ml  "$ROOT/data/hh_rlhf/hh_rlhf_5k_malayalam.jsonl" \
    --output   "$ROOT/checkpoints/dpo_multilingual" \
    --epochs 1 --batch 2 --grad-accum 8 --lr 5e-5 --max-len 1024 \
    --logging-steps 20

echo "[1/3] Training done. Checkpoint: $ROOT/checkpoints/dpo_multilingual"

# ── Step 2: Start vLLM with DPO checkpoint ─────────────────────────────────────
echo ""
echo "[2/3] Starting vLLM with DPO checkpoint..."
export CUDA_VISIBLE_DEVICES=$GPUS
export HTTP_PROXY=http://127.0.0.1:3128
export HTTPS_PROXY=http://127.0.0.1:3128
export VLLM_USE_V1=1

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$ROOT/checkpoints/dpo_multilingual" \
    --served-model-name dpo-8b \
    --tensor-parallel-size $TP \
    --port $PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 &
VLLM_PID=$!

echo "Waiting for vLLM to be ready (PID $VLLM_PID)..."
for i in $(seq 1 60); do
    curl -s http://localhost:$PORT/health > /dev/null 2>&1 && break
    sleep 5
done
echo "vLLM ready."

# ── Step 3: Eval on all 3 languages ───────────────────────────────────────────
echo ""
echo "[3/3] Running HHH eval on DPO model..."

for LANG_TAG in english hindi malayalam; do
    if [ "$LANG_TAG" = "english" ]; then
        DATA="$ROOT/data/hhh_alignment/english.jsonl"
        OUT="$LOG_DIR/hhh_dpo_${LANG_TAG}.json"
    elif [ "$LANG_TAG" = "hindi" ]; then
        DATA="$ROOT/data/hhh_alignment/hindi_gemma3_27b.jsonl"
        OUT="$LOG_DIR/hhh_dpo_${LANG_TAG}.json"
    else
        DATA="$ROOT/data/hhh_alignment/malayalam_gemma3_27b.jsonl"
        OUT="$LOG_DIR/hhh_dpo_${LANG_TAG}.json"
    fi
    echo "  Evaluating $LANG_TAG..."
    $PYTHON "$ROOT/eval_hhh.py" \
        --model dpo-8b \
        --base-url "http://localhost:$PORT/v1" \
        --data "$DATA" \
        --batch-size 64 \
        --max-tokens-think 8000 \
        --output "$OUT"
done

# ── Stop vLLM ──────────────────────────────────────────────────────────────────
kill $VLLM_PID 2>/dev/null || true

echo ""
echo "=== Pipeline complete. Results in $LOG_DIR ==="
echo "Before (8B baseline): EN=90.95%  HI=86.70%  ML=79.09%"
for LANG_TAG in english hindi malayalam; do
    OUT="$LOG_DIR/hhh_dpo_${LANG_TAG}.json"
    if [ -f "$OUT" ]; then
        ACC=$($PYTHON -c "import json; d=json.load(open('$OUT')); print(f\"{d['modes']['zero-shot + thinking']['accuracy']:.2f}%\")")
        echo "After  (DPO-8B):     $LANG_TAG=$ACC"
    fi
done
