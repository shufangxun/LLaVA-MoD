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
MLLM='./checkpoints/llavaqwen-2-0.5b-finetune'
VISION_ENCODER='openai/clip-vit-large-patch14-336'

# Output dir
OUTPUT_DIR='./checkpoints/llavaqwen-2-0.5b-finetune-moe'

# MoE config
MOE_MODE="sparse"
MOE_ENABLE=True
MOE_FINETUNE=False
NUM_EXPERTS=4
TOP_K_EXPERTS=2
USE_RESIDUAL=False
ROUTER_AUX_LOSS_COEF=0.01
CAPACITY_FACTOR=1.5

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed llavamod/train/train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --moe_enable ${MOE_ENABLE} --moe_finetune ${MOE_FINETUNE} --num_experts ${NUM_EXPERTS} --top_k_experts ${TOP_K_EXPERTS} --capacity_factor ${CAPACITY_FACTOR} \
    --moe_mode ${MOE_MODE} --use_residual ${USE_RESIDUAL} --router_aux_loss_coef ${ROUTER_AUX_LOSS_COEF} \
    --train_modules mlp.w1 mlp.w2 mlp.c_proj wg \
    --model_name_or_path ${MLLM} \
    --version qwen \
    --data_path ${JSON_FILE} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${VISION_ENCODER} \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --attn_implementation sdpa \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
