#!/bin/bash

# WANDB
WANDB_API_KEY='your_wandb_api_key'
export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_MODE=online
wandb login ${WANDB_API_KEY}
wandb online

# Deepspeed config
DEEPSPEED_CONFIG='./llavamod/config/dpconfig/zero2.json'

# Dataset
JSON_FILE='your_json_file'
IMAGE_FOLDER='your_image_folder'

# Pretrain models
LLM='Qwen/Qwen2-0.5B'
VISION_ENCODER='openai/clip-vit-large-patch14-336'

# Output dir
OUTPUT_DIR='./checkpoints/llavaqwen-2-0.5b-pretrain'

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed llavamod/train/train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${LLM} \
    --version qwen \
    --data_path ${JSON_FILE} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${VISION_ENCODER} \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --attn_implementation sdpa \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb


