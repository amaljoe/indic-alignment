#!/bin/bash
# Start vLLM inside apptainer (required on this RHEL node — host gcc has no C headers)
# Usage: bash scripts/start_vllm.sh [model_id] [port] [tp_size]
#
# The apptainer Ubuntu container has gcc + stdlib.h needed by triton.
# Triton cache lives at ~/.triton/cache/ and persists across runs.

MODEL=${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
PORT=${2:-8002}
TP=${3:-2}
SERVED_NAME=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
LOG="$HOME/vllm_${SERVED_NAME}.log"

echo "Starting vLLM: $MODEL on port $PORT (tp=$TP)"
echo "Log: $LOG"

apptainer exec --nv ~/apptainer-images/cuda-custom-amal_latest.sif bash -c "
  source ~/.bashrc 2>/dev/null
  micromamba activate /dev/shm/qwen35
  export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6
  export HTTP_PROXY=http://127.0.0.1:3128
  export HTTPS_PROXY=http://127.0.0.1:3128
  python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name $SERVED_NAME \
    --tensor-parallel-size $TP \
    --port $PORT \
    --host 0.0.0.0 \
    --max-model-len 8192
" 2>&1 | tee "$LOG"
