## Preliminary
### Download Pretrained Checkpoints
We use [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) as the vision encoder for both teacher and student models. Additionally, we use [Qwen-1.5](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524) / [Qwen-2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f) of different sizes respectively as the LLM for the teacher and student models. These pretrained checkpoints can be downloaded from HuggingFace.

### Prepare Teacher Model
We follow the approach of [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) to train the teacher model, replacing Vicuna-1.5-7B with Qwen-2-7B, while keeping the training dataset and strategy unchanged. 

## Training 
The training of LLaVA-MoD comprises three stages:
- Adaptor Initialization: 0.6 million general captioning samples are employed to bridge the gap between visual and language modalities.
- Mimic Distillation: 
  - Dense-to-Dense Distillation: 2.4 million general captioning and conversation samples are utilized to distill general knowledge.
  - Dense-to-Sparse Distillation: 1.4 million multi-task data, including VQA, documents, science, and OCR, are used to distill specialized knowledge.
- Preference Distillation tuning stage: 80,000 preference data samples are utilized to distill preference knowledge.

### Adaptor Initialization
- first, download the caption dataset [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) 
- then run the following scripts:
```shell
bash shells/train/qwen/pretrain.sh
```

### Mimic Distillation
In this stage, we initially conduct Dense-to-Dense Distillation on the dense student model. Subsequently, we up-cycle the student model from dense to sparse and conduct Dense-to-Sparse Distillation.

#### Dense-to-Dense Distillation 
- first, download general caption datasets ([ShareGPT4V-Captioner](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json) and [ALLaVA-Caption-LAION-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/blob/main/allava_laion/ALLaVA-Caption-LAION-4V.json)) and general conversation datasets ([SViT](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning), [LVIS](https://github.com/X2FD/LVIS-INSTRUCT4V), [LRV](https://github.com/FuxiaoLiu/LRV-Instruction), [MIMIC-IT](https://github.com/Luodian/Otter)). The general datasets have also been packaged and can be downloaded from [MoE-LLaVA](https://huggingface.co/datasets/LanguageBind/MoE-LLaVA).
- then, set the distillation and model configuration:
```python
# KD config
POLICY_MODEL_TYPE='dense'
REF_MODEL_TYPE='dense'
LOSS_TYPE='only_kd'  # kd_lm | only_kd
DISTILL_ALL_TOKENS=False  # False: only response, True: multimodal instruction + response

# MoE config
MOE_LOSS_ENABLE=False
MOE_ENABLE=False
MOE_FINETUNE=False
MOE_MODE="sparse"
NUM_EXPERTS=4
TOP_K_EXPERTS=2
USE_RESIDUAL=False
ROUTER_AUX_LOSS_COEF=0.01
CAPACITY_FACTOR=1.5
```
- finally, run the following scripts:
```shell
bash shells/train/qwen/dense2dense_distillation.sh
```

#### Dense-to-Sparse Distillation
- first, download multi-task datasets ([Text-VQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), 
  [IConQA](https://drive.google.com/file/d/1Xqdt1zMcMZU5N_u1SAIjk-UAclriynGx/edit), [SQA](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev), [SBU](https://huggingface.co/datasets/sbu_captions), follow [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md) to download images from:
  [LAION-CC-SBU-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip), [COCO](http://images.cocodataset.org/zips/train2017.zip), [WebData](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing), [SAM](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link), [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), [VisualGnome](https://cs.stanford.edu/people/rak248/VG_100K_2) ([Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)), follow [InternVL](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data) to download [DVQA](https://github.com/kushalkafle/DVQA_dataset), [ChartQA](https://github.com/vis-nlp/ChartQA), [AI2D](https://allenai.org/data/diagrams), [DocVQA](https://www.docvqa.org/datasets), [GeoQA+](https://github.com/SCNU203/GeoQA-Plus), and [SynthDoG-EN](https://huggingface.co/datasets/naver-clova-ix/synthdog-en)). The json files have also been packaged and can be downloaded from [MobileVLM](https://huggingface.co/datasets/mtgv/MobileVLM_V2_FT_Mix2M) and [InternVL](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data).
- then, set the distillation and model configuration:
```python
# KD config
POLICY_MODEL_TYPE='dense'
REF_MODEL_TYPE='dense'
LOSS_TYPE='only_kd'  # kd_lm | only_kd
DISTILL_ALL_TOKENS=False  # False: only response, True: multimodal instruction + response

# MoE config
MOE_LOSS_ENABLE=False
MOE_ENABLE=False
MOE_FINETUNE=False
MOE_MODE="sparse"
NUM_EXPERTS=4
TOP_K_EXPERTS=2
USE_RESIDUAL=False
ROUTER_AUX_LOSS_COEF=0.01
CAPACITY_FACTOR=1.5
```
- finally, run the following scripts:
```shell
bash shells/train/qwen/dense2sparse_distillation.sh
```


### Preference Distillation
- first, download preference dataset from [RLAIF-V](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset).
- then, set the distillation and model configuration:
```python
# KD config
POLICY_MODEL_TYPE='sparse'
REF_MODEL_TYPE='dense'
LOSS_TYPE='kto_pair'  # kto_pair | sigmoid
DISTILL_ALL_TOKENS=False  # False: only response, True: multimodal instruction + response


# MoE config
MOE_LOSS_ENABLE=True
MOE_ENABLE=True
MOE_FINETUNE=True
MOE_MODE="sparse"
NUM_EXPERTS=4
TOP_K_EXPERTS=2
USE_RESIDUAL=False
ROUTER_AUX_LOSS_COEF=0.01
CAPACITY_FACTOR=1.5
```
- finally, run the following scripts:
```shell
bash shells/train/qwen/preference_distillation.sh
```

## Evaluation
We follow [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) to evaluate on comprehension benchmarks (TextVQA, GQA, ScienceQA, VizWiz, MME, MMBench) and [RLAIF-V](https://github.com/RLHF-V/RLAIF-V) to evaluate on hallucination benchmarks (MMHal Bench, POPE and Object HalBench). Please refer to these resources to organize the evaluation datasets. All the evaluation scripts are located under `shells/eval`. Here is an example for MMBench.
```shell
#!/bin/bash
MODEL_NAME='your_model_name'
MODEL_PATH='your_model_path'

CONV="qwen"
SPLIT="mmbench_dev_en_20231003"
EVAL="benchmark"

deepspeed --include localhost:0 --master_port 20029 llavamod/eval/model_vqa_mmbench.py \
     --model-path ${MODEL_PATH} \
     --question-file ${EVAL}/mmbench/$SPLIT.tsv \
     --answers-file ${EVAL}/mmbench/answers/$SPLIT/${MODEL_NAME}.jsonl \
     --single-pred-prompt \
     --temperature 0 \
     --conv-mode ${CONV}

mkdir -p ${EVAL}/mmbench/answers_upload/$SPLIT

python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file ${EVAL}/mmbench/$SPLIT.tsv \
    --result-dir ${EVAL}/mmbench/answers/$SPLIT \
    --upload-dir ${EVAL}/mmbench/answers_upload/$SPLIT \
    --experiment ${MODEL_NAME}
```


