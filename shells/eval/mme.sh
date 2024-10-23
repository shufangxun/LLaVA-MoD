#!/bin/bash

MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"
EVAL="benchmark"
IMAGE_FOLDER="your_image_folder"

deepspeed --include localhost:0 --master_port 20018 llavamod/eval/model_vqa_loader.py \
     --model-path ${MODEL_PATH} \
     --question-file ${EVAL}/MME/llava_mme.jsonl \
     --image-folder ${IMAGE_FOLDER} \
     --answers-file ${EVAL}/MME/answers/${MODEL_NAME}.jsonl \
     --temperature 0 \
     --num_beams 1 \
     --conv-mode ${CONV}


cd ${EVAL}/MME

python convert_answer_to_mme.py --experiment ${MODEL_NAME}


cd eval_tool

python calculation.py --results_dir answers/${MODEL_NAME}

