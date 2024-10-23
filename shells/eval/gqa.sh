#!/bin/bash

GPULIST=(0 1 2 3)
CHUNKS=${#GPULIST[@]}

MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"
EVAL="benchmark"

SPLIT="llava_gqa_testdev_balanced"
GQADIR="${EVAL}/gqa/data"
IMAGE_FOLDER="your_image_folder"


for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) llavamod/eval/model_vqa_loader.py \
        --model-path ${MODEL_PATH} \
        --question-file ${EVAL}/gqa/$SPLIT.jsonl \
        --image-folder ${IMAGE_FOLDER} \
        --answers-file ${EVAL}/gqa/answers/$SPLIT/${MODEL_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV} &
done

wait

output_file=${EVAL}/gqa/answers/$SPLIT/${MODEL_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/gqa/answers/$SPLIT/${MODEL_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $GQADIR/$SPLIT/${MODEL_NAME}
python3 scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/$SPLIT/${MODEL_NAME}/testdev_balanced_predictions.json

# cd $GQADIR
python3 llavamod/eval/eval_gqa_1.py \
      --tier ${EVAL}/gqa/data/${SPLIT}/${MODEL_NAME}/testdev_balanced \
      --questions ${EVAL}/gqa/data/testdev_balanced_questions.json

                        