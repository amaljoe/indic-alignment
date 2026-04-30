#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,2
export VLLM_USE_V1=1
export HTTP_PROXY=http://127.0.0.1:3128
export HTTPS_PROXY=http://127.0.0.1:3128
export LD_PRELOAD=/dev/shm/vllm/lib/libstdc++.so.6
exec /dev/shm/vllm/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --tensor-parallel-size 2 \
    --port 8003 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --reasoning-parser qwen3
