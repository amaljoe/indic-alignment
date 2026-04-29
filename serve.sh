#!/bin/bash
# Student vLLM server — runs persistently for all eval phases.
# Requires --enable-lora for hot-swappable LoRA adapters.
# Usage: bash serve.sh [port=8002] [gpus=0,1] [tp=2] [max_len=8192]
set -e

MODEL=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
PORT=${1:-8002}
GPUS=${2:-0,1}
TP=${3:-2}
MAX_LEN=${4:-8192}
LOG="$HOME/vllm_student.log"

echo "Starting student vLLM: $MODEL"
echo "  port=$PORT  gpus=$GPUS  tp=$TP  max_len=$MAX_LEN"
echo "  log: $LOG"

apptainer exec --nv ~/images/cuda-custom-amal_latest.sif bash -c "
  export CUDA_VISIBLE_DEVICES=$GPUS
  export VLLM_USE_V1=1
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
  export HTTP_PROXY=http://127.0.0.1:3128
  export HTTPS_PROXY=http://127.0.0.1:3128
  /home/compiling-ganesh/24m0797/envs/vllm/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name deepseek-r1-8b \
    --tensor-parallel-size $TP \
    --port $PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len $MAX_LEN \
    --enable-lora \
    --max-lora-rank 32 \
    --reasoning-parser qwen3
" 2>&1 | tee "$LOG"
