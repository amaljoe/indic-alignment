#!/bin/bash
# Start vLLM inside apptainer using the home-dir env at ~/envs/vllm.
# Usage: bash scripts/start_vllm_v2.sh <model_id> <port> <tp_size> <gpus> <served_name>
#   gpus: e.g. "0,1" or "2,3"
set -e

MODEL=${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
PORT=${2:-8002}
TP=${3:-2}
GPUS=${4:-0,1}
SERVED_NAME=${5:-deepseek-r1}
MAX_LEN=${MAX_LEN:-8192}

LOG="$HOME/vllm_${SERVED_NAME}.log"

echo "Starting vLLM: $MODEL on port $PORT (tp=$TP, GPUs=$GPUS, name=$SERVED_NAME)"
echo "Log: $LOG"

apptainer exec --nv --bind /scratch:/scratch ~/images/cuda-custom-amal_latest.sif bash -c "
  export CUDA_VISIBLE_DEVICES=$GPUS
  export HTTP_PROXY=http://127.0.0.1:3128
  export HTTPS_PROXY=http://127.0.0.1:3128
  export VLLM_USE_V1=1
  /home/compiling-ganesh/24m0797/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name $SERVED_NAME \
    --tensor-parallel-size $TP \
    --port $PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len $MAX_LEN
" 2>&1 | tee "$LOG"
