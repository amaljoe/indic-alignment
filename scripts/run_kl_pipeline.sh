#!/bin/bash
# KL-distillation pipeline: generate soft labels → train → eval
#
# Usage:
#   bash scripts/run_kl_pipeline.sh [--alpha 0.5] [--temperature 2.0]
#                                    [--skip-datagen] [--skip-train] [--skip-eval]
#
# Environment: uses /dev/shm/qwen35 Python (torch 2.10, transformers 5.6.2)
# Teacher model: Qwen/Qwen3-30B-A3B (loaded via HuggingFace, device_map=auto)
# Student model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

set -euo pipefail

PYTHON=/dev/shm/qwen35/bin/python
ACCELERATE=/dev/shm/qwen35/bin/accelerate
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

# ── Default hyperparameters ────────────────────────────────────────────────────
ALPHA=0.5
TEMPERATURE=2.0
EPOCHS=3
LR=2e-5
BATCH=2
GRAD_ACCUM=8
MAX_SEQ=1024

STUDENT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEACHER_MODEL="Qwen/Qwen3-30B-A3B"
OUTPUT_DIR="finetune/checkpoints_kl"
RESULTS_DIR="results/kl"

SKIP_DATAGEN=0
SKIP_TRAIN=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --alpha)        ALPHA="$2";        shift 2 ;;
        --temperature)  TEMPERATURE="$2";  shift 2 ;;
        --epochs)       EPOCHS="$2";       shift 2 ;;
        --lr)           LR="$2";           shift 2 ;;
        --output)       OUTPUT_DIR="$2";   shift 2 ;;
        --skip-datagen) SKIP_DATAGEN=1;    shift   ;;
        --skip-train)   SKIP_TRAIN=1;      shift   ;;
        --skip-eval)    SKIP_EVAL=1;       shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

FINAL_MODEL="$OUTPUT_DIR/final"
mkdir -p "$RESULTS_DIR"

LOG_TAG="alpha${ALPHA}_T${TEMPERATURE}"
RESULTS_FILE="$RESULTS_DIR/normad_${LOG_TAG}.json"

# ── Activate proxy ─────────────────────────────────────────────────────────────
source ~/proxy-setup/scripts/proxy_env.sh 2>/dev/null || true
export HTTP_PROXY=http://127.0.0.1:3128
export HTTPS_PROXY=http://127.0.0.1:3128
export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6

echo "============================================================"
echo " KL Distillation Pipeline"
echo "  alpha=$ALPHA  temperature=$TEMPERATURE  epochs=$EPOCHS"
echo "  output=$OUTPUT_DIR"
echo "============================================================"

# ── Step 1: Kill any running vLLM ─────────────────────────────────────────────
echo ""
echo "[Step 1] Stopping any running vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 5

# ── Step 2: Generate soft labels ──────────────────────────────────────────────
if [[ "$SKIP_DATAGEN" -eq 0 ]]; then
    echo ""
    echo "[Step 2] Generating soft labels with $TEACHER_MODEL..."
    $PYTHON finetune/generate_soft_labels.py \
        --model "$TEACHER_MODEL" \
        --sources normad milu_en milu_hi bhed globalopinion \
        --filter-correct
    echo "[Step 2] Done."
else
    echo "[Step 2] Skipped (--skip-datagen)."
fi

# ── Step 3: Train student with KL distillation ────────────────────────────────
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    echo ""
    echo "[Step 3] Training student with KL distillation..."

    if [[ -f finetune/accelerate_config.yaml ]]; then
        $ACCELERATE launch \
            --config_file finetune/accelerate_config.yaml \
            finetune/train_kl.py \
            --model "$STUDENT_MODEL" \
            --output "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --batch-size "$BATCH" \
            --grad-accum "$GRAD_ACCUM" \
            --max-seq-len "$MAX_SEQ" \
            --alpha "$ALPHA" \
            --temperature "$TEMPERATURE"
    else
        $PYTHON finetune/train_kl.py \
            --model "$STUDENT_MODEL" \
            --output "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --batch-size "$BATCH" \
            --grad-accum "$GRAD_ACCUM" \
            --max-seq-len "$MAX_SEQ" \
            --alpha "$ALPHA" \
            --temperature "$TEMPERATURE"
    fi
    echo "[Step 3] Done. Model saved to $FINAL_MODEL"
else
    echo "[Step 3] Skipped (--skip-train)."
fi

# ── Step 4: Evaluate ──────────────────────────────────────────────────────────
if [[ "$SKIP_EVAL" -eq 0 ]]; then
    echo ""
    echo "[Step 4] Starting vLLM for evaluation..."

    # Start vLLM in background
    MODEL_FOR_VLLM="${FINAL_MODEL:-$STUDENT_MODEL}"
    bash scripts/start_vllm.sh "$MODEL_FOR_VLLM" 8002 2 &
    VLLM_PID=$!

    # Wait for server to come up
    echo "Waiting for vLLM on port 8002..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8002/v1/models > /dev/null 2>&1; then
            echo "vLLM ready."
            break
        fi
        sleep 5
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM process died — check ~/vllm_final.log"
            exit 1
        fi
    done

    SERVED_NAME=$(basename "$MODEL_FOR_VLLM" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')

    # NormAd — test countries only (Nepal + Sri Lanka held out from training)
    echo ""
    echo "[Step 4a] NormAd evaluation (test set: nepal + sri_lanka)..."
    $PYTHON eval_normad.py \
        --model "$SERVED_NAME" \
        --base-url http://localhost:8002/v1 \
        --countries nepal sri_lanka \
        --output "$RESULTS_FILE" \
        --few-shot-n 3 \
        --batch-size 16

    # BhED — full eval (train/test from CSV; quick proxy for debiasing)
    echo ""
    echo "[Step 4b] BhED evaluation..."
    $PYTHON eval_bhed.py \
        --model "$SERVED_NAME" \
        --base-url http://localhost:8002/v1 \
        --output "$RESULTS_DIR/bhed_${LOG_TAG}.json"

    echo ""
    echo "[Step 4] Evaluation done.  Results in $RESULTS_DIR/"
    echo "  NormAd: $RESULTS_FILE"
    echo "  BhED:   $RESULTS_DIR/bhed_${LOG_TAG}.json"

    # Kill vLLM when done
    kill $VLLM_PID 2>/dev/null || true
fi

echo ""
echo "============================================================"
echo " Pipeline complete."
echo "============================================================"
