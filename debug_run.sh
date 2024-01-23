#!/bin/bash
set -e

DATASET=medmcqa
HF_NAME="01-ai/Yi-34B-Chat"
SHORT_NAME=yi
OUT_PATH="/weka/home-griffin/${SHORT_NAME}_${DATASET}.jsonl"

echo "Running ${DATASET} and saving to ${OUT_PATH}"
python3 -m lm_eval \
    --output_path $OUT_PATH \
    --log_samples \
    --model vllm \
    --model_args "pretrained=${HF_NAME},trust_remote_code=True,dtype=auto,tensor_parallel_size=8,gpu_memory_utilization=0.8" \
    --device cuda \
    --batch_size auto \
    --tasks ${DATASET}_medprompt
