#!/bin/bash

MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"

EVAL="benchmark"
IMAGE_FOLDER="your_image_folder"

deepspeed --include localhost:0 --master_port 20013 llavamod/eval/model_vqa_loader.py \
    --model-path ${MODEL_PATH} \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --image-folder ${IMAGE_FOLDER} \
    --answers-file ${EVAL}/pope/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 llavamod/eval/eval_pope.py \
    --annotation-dir ${EVAL}/pope/coco \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --result-file ${EVAL}/pope/answers/${MODEL_NAME}.jsonl
