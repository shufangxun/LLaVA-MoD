#!/bin/bash

MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"
EVAL="benchmark"

deepspeed --include localhost:0 --master_port 20017 llavamod/eval/model_vqa_science.py \
    --model-path ${MODEL_PATH} \
    --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
    --image-folder ${EVAL}/scienceqa/images/test \
    --answers-file ${EVAL}/scienceqa/answers/${MODEL_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV}

python3 llavamod/eval/eval_science_qa.py \
    --base-dir ${EVAL}/scienceqa \
    --result-file ${EVAL}/scienceqa/answers/${MODEL_NAME}.jsonl \
    --output-file ${EVAL}/scienceqa/answers/${MODEL_NAME}_output.jsonl \
    --output-result ${EVAL}/scienceqa/answers/${MODEL_NAME}_result.json