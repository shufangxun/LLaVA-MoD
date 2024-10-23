#!/bin/bash

MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"
EVAL="benchmark"

deepspeed --include localhost:0 --master_port 20016 llavamod/eval/model_vqa_loader.py \
    --model-path ${MODEL_PATH} \
    --question-file ${EVAL}/vizwiz/llava_test.jsonl \
    --image-folder ${EVAL}/vizwiz/test \
    --answers-file ${EVAL}/vizwiz/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${EVAL}/vizwiz/llava_test.jsonl \
    --result-file ${EVAL}/vizwiz/answers/${MODEL_NAME}.jsonl \
    --result-upload-file ${EVAL}/vizwiz/answers_upload/${MODEL_NAME}.json