#!/bin/bash


model_path='your_model'
image_file='llavamod/serve/examples/extreme_ironing.jpg'

# use qwen
deepspeed --include localhost:0 --master_port 20019 llavamod/serve/cli.py \
  --model-path ${model_path} --image-file ${image_file}
