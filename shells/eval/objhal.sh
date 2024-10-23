#!/bin/bash

MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"
EVAL="benchmark"
question_file='your_question_file'

openai_key='yourt_openai_key'

deepspeed --include localhost:0 --master_port 20015 llavamod/eval/model_vqa_objhal.py \
    --model-path ${MODEL_PATH} \
    --question-file ${question_file} \
    --answers-file ${EVAL}/objhal/answers/${MODEL_NAME}.jsonl \
    --temperature 0.0 \
    --conv-mode ${CONV}

coco_annotation_path=${EVAL}/objhal/annotation

python3 llavamod/eval/eval_gpt_objhal.py \
    --coco_path ${coco_annotation_path} \
    --cap_file ${EVAL}/objhal/answers/${MODEL_NAME}.jsonl \
    --org_folder ${question_file} \
    --use_gpt \
    --openai_key ${openai_key}
