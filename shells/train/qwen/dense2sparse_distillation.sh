#!/bin/bash

# WANDB
WANDB_API_KEY='your_wandb_api_key'
export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_MODE=online
wandb login ${WANDB_API_KEY}
wandb online

# Deepspeed config
DEEPSPEED_CONFIG='./llavamod/config/dpconfig/zero2_offload.json'

# Dataset
JSON_FILE='your_json_file'
IMAGE_FOLDER='your_image_folder'

# Teacher
REF_MLLM='your_reference_mllm'

# Student
POLICY_MLLM='./checkpoints/llavaqwen-2-0.5b-d2d'

# Vision encoder
VISION_ENCODER='openai/clip-vit-large-patch14-336'

# KD config
POLICY_MODEL_TYPE='sparse'
REF_MODEL_TYPE='dense'
LOSS_TYPE='kd_lm'  # kd_lm | only_kd
DISTILL_ALL_TOKENS=False  # False: only response, True: multimodal instruction + response


# MoE config
MOE_LOSS_ENABLE=True
MOE_ENABLE=True
MOE_FINETUNE=False
MOE_MODE="sparse"
NUM_EXPERTS=4
TOP_K_EXPERTS=2
USE_RESIDUAL=False
ROUTER_AUX_LOSS_COEF=0.01
CAPACITY_FACTOR=1.5


# Output dir
OUTPUT_DIR='./checkpoints/llavaqwen-2-0.5b-d2s'

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed llavamod/train/align_train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --ref_model_name_or_path ${REF_MLLM} \
    --policy_model_name_or_path ${POLICY_MLLM} \
    --policy_model_type ${POLICY_MODEL_TYPE} --ref_model_type ${REF_MODEL_TYPE} --loss_type ${LOSS_TYPE} \
    --moe_loss_enable ${MOE_LOSS_ENABLE} --moe_enable ${MOE_ENABLE} --moe_finetune ${MOE_FINETUNE} \
    --num_experts ${NUM_EXPERTS} --top_k_experts ${TOP_K_EXPERTS} --capacity_factor ${CAPACITY_FACTOR} \
    --moe_mode ${MOE_MODE} --use_residual ${USE_RESIDUAL} --router_aux_loss_coef ${ROUTER_AUX_LOSS_COEF} \
    --train_modules mlp.gate_proj mlp.up_proj mlp.down_proj wg \
    --distill_all_tokens ${DISTILL_ALL_TOKENS} \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --attn_implementation sdpa \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
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

