#!/bin/bash
# Wait for both translations to finish, then:
#  1. Stop Gemma 27B
#  2. DPO LoRA training (4 GPUs)
#  3. Start vLLM with DPO checkpoint
#  4. Eval on en/hi/ml (thinking-only)
#  5. Print comparison table
set -e

PYTHON=/home/compiling-ganesh/24m0797/envs/vllm/bin/python
ROOT=/home/compiling-ganesh/24m0797/workspace/indic-alignment
MODEL=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
CKPT=$ROOT/checkpoints/dpo_multilingual
PORT=8003

echo "=== post_translate_pipeline.sh started at $(date) ==="

# ── Wait for translations ──────────────────────────────────────────────────────
echo "Waiting for translation files..."
while true; do
    HI_N=0; ML_N=0
    if [ -f "$ROOT/data/hh_rlhf/hh_rlhf_5k_hindi.jsonl" ]; then
        HI_N=$(wc -l < "$ROOT/data/hh_rlhf/hh_rlhf_5k_hindi.jsonl")
    fi
    if [ -f "$ROOT/data/hh_rlhf/hh_rlhf_5k_malayalam.jsonl" ]; then
        ML_N=$(wc -l < "$ROOT/data/hh_rlhf/hh_rlhf_5k_malayalam.jsonl")
    fi
    echo "  $(date +%H:%M) Hindi=$HI_N/5000, Malayalam=$ML_N/5000"
    if [ "$HI_N" -ge 4990 ] && [ "$ML_N" -ge 4990 ]; then
        break
    fi
    sleep 120
done
echo "Both translations complete."

# ── Guard: skip if checkpoint already exists ──────────────────────────────────
if [ -f "$CKPT/adapter_config.json" ] || [ -f "$CKPT/config.json" ]; then
    echo "Checkpoint already exists at $CKPT — skipping training."
    exit 0
fi

# ── Stop Gemma 27B (kill vLLM server on port 8003) ────────────────────────────
echo "Stopping Gemma 27B..."
pkill -f "port 8003" 2>/dev/null || true
pkill -f "gemma3-27b" 2>/dev/null || true
sleep 10
echo "Gemma 27B stopped."

# ── DPO Training ──────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Starting DPO LoRA training at $(date)..."
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 $PYTHON "$ROOT/scripts/train_dpo_lora.py" \
    --model "$MODEL" \
    --data-en  "$ROOT/data/hh_rlhf/hh_rlhf_5k_en.jsonl" \
    --data-hi  "$ROOT/data/hh_rlhf/hh_rlhf_5k_hindi.jsonl" \
    --data-ml  "$ROOT/data/hh_rlhf/hh_rlhf_5k_malayalam.jsonl" \
    --output   "$CKPT" \
    --epochs 1 --batch 2 --grad-accum 16 --lr 5e-5 --max-len 1024 \
    --logging-steps 20
echo "[1/3] Training done at $(date)."

# ── Start vLLM with DPO checkpoint ────────────────────────────────────────────
echo ""
echo "[2/3] Starting vLLM with DPO checkpoint..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HTTP_PROXY=http://127.0.0.1:3128
export HTTPS_PROXY=http://127.0.0.1:3128
export VLLM_USE_V1=1

nohup $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" \
    --served-model-name dpo-8b \
    --tensor-parallel-size 4 \
    --port $PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 > ~/vllm_dpo-8b.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    curl -s http://localhost:$PORT/health > /dev/null 2>&1 && break
    sleep 5
done
echo "vLLM ready."

# ── Eval ──────────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Running HHH eval..."
for LANG_TAG in english hindi malayalam; do
    if [ "$LANG_TAG" = "english" ]; then
        DATA="$ROOT/data/hhh_alignment/english.jsonl"
    elif [ "$LANG_TAG" = "hindi" ]; then
        DATA="$ROOT/data/hhh_alignment/hindi_gemma3_27b.jsonl"
    else
        DATA="$ROOT/data/hhh_alignment/malayalam_gemma3_27b.jsonl"
    fi
    OUT="$ROOT/results/hhh_dpo_${LANG_TAG}.json"
    echo "  Evaluating $LANG_TAG -> $OUT"
    $PYTHON "$ROOT/eval_hhh.py" \
        --model dpo-8b \
        --base-url "http://localhost:$PORT/v1" \
        --data "$DATA" \
        --batch-size 64 \
        --max-tokens-think 8000 \
        --output "$OUT"
done

kill $VLLM_PID 2>/dev/null || true

# ── Print comparison ──────────────────────────────────────────────────────────
echo ""
echo "=== Results ==="
echo "Baseline 8B (before DPO): EN=90.95%  HI=86.70%  ML=79.09%"
for LANG_TAG in english hindi malayalam; do
    OUT="$ROOT/results/hhh_dpo_${LANG_TAG}.json"
    if [ -f "$OUT" ]; then
        ACC=$($PYTHON -c "import json; d=json.load(open('$OUT')); print(f\"{d['modes']['zero-shot + thinking']['accuracy']:.2f}%\")" 2>/dev/null || echo "N/A")
        echo "DPO-8B: $LANG_TAG=$ACC"
    fi
done
echo "Pipeline complete at $(date)"
