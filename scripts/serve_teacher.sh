#!/bin/bash
# Teacher vLLM server — Gemma 3 27B for distillation & translation.
# Usage: bash serve_teacher.sh [port=8003] [gpus=2,3] [tp=2] [max_len=8192]
set -e

MODEL=google/gemma-3-27b-it
PORT=${1:-8003}
GPUS=${2:-2,3}
TP=${3:-2}
MAX_LEN=${4:-8192}
LOG="$HOME/vllm_teacher.log"

echo "Starting Gemma 3 27B teacher: $MODEL"
echo "  port=$PORT  gpus=$GPUS  tp=$TP  max_len=$MAX_LEN"
echo "  log: $LOG"

apptainer exec --nv ~/images/cuda-custom-amal_latest.sif bash -c "
  export CUDA_VISIBLE_DEVICES=$GPUS
  export VLLM_USE_V1=1
  export HTTP_PROXY=http://127.0.0.1:3128
  export HTTPS_PROXY=http://127.0.0.1:3128
  /home/compiling-ganesh/24m0797/envs/vllm/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name gemma3-27b \
    --tensor-parallel-size $TP \
    --port $PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.92 \
    --max-model-len $MAX_LEN
" 2>&1 | tee "$LOG"
