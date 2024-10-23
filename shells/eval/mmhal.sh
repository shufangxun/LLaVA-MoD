#!/bin/bash

MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"
EVAL="benchmark"
IMAGE_FOLDER="your_image_folder"
QUESTION_FILE="your_question_file"
# TEST_PROMPT="\nAnswer the question using a single word or phrase."

deepspeed --include localhost:0 --master_port 20012 llavamod/eval/model_vqa_mmhal.py  \
    --model-path ${MODEL_PATH} \
    --question-file ${QUESTION_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --answers-file ${EVAL}/mmhal/answers/${MODEL_NAME}.jsonl \
    --temperature 0.0 \
    --max_new_tokens 1024 \
    --conv-mode ${CONV} \

python3 llavamod/eval/eval_gpt_mmhal.py \
 --response ${EVAL}/mmhal/answers/${MODEL_NAME}.jsonl \
 --evaluation ${EVAL}/mmhal/answers/${MODEL_NAME}_sum.jsonl \
