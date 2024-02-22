#!/bin/bash
set -e

DATASET=$1
LIMIT=$2
HF_NAME="01-ai/Yi-34B"
SHORT_NAME=yi
OUT_PATH="/weka/home-griffin/harness_results/${SHORT_NAME}_${DATASET}.jsonl"

echo "Running ${DATASET} for ${LIMIT} examples and saving to ${OUT_PATH}"
python3 -m lm_eval \
    --output_path $OUT_PATH \
    --log_samples \
    --limit $2 \
    --model vllm \
    --model_args "pretrained=${HF_NAME},trust_remote_code=True,dtype=auto,tensor_parallel_size=8" \
    --device cuda \
    --batch_size 8 \
    --tasks ${DATASET}_medprompt
