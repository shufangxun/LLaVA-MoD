## Inference

We provide commandline inference and batch inference scripts.

### CLI Inference
```Shell
deepspeed --include localhost:0 --master_port 20019 llavamod/serve/cli.py \
  --model-path ${MODEL_PATH} --image-file ${IMAGE_FILE}
```
### Batch Inference
```Shell
deepspeed --master_port 20014 llavamod/eval/model_vqa.py \
    --model-path ${MODEL_PATH} \
    --question-file ${QUESTION_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --answers-file ${ANSWER_FILE} \
    --temperature 0.0 \
    --conv-mode qwen
```