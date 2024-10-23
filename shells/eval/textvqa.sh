#!/bin/bash

MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"
EVAL="benchmark"

deepspeed --include localhost:0 --master_port 20030 llavamod/eval/model_vqa_loader.py \
    --model-path ${MODEL_PATH} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 -m llavamod.eval.eval_textvqa \
    --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${EVAL}/textvqa/answers/${MODEL_NAME}.jsonl