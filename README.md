# <div align="center">LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation<div>

<div align="center">
<!-- [![arXiv](https://img.shields.io/badge/Arxiv-2408.15881-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2408.15881) -->
  <a href="[https://arxiv.org/abs/2408.15881](https://arxiv.org/abs/2408.15881)"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:LLaVA-MOD&color=red&logo=arxiv"></a>
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/shufangxun/LLaVA-MoD/blob/main/LICENSE)
</div>

---

## 📢 News
- Jan 23, 2025: 🔥 LLaVA-MoD is accepted by ICLR 2025.
- Dec 24, 2024: 🏋️‍♀️ Training and evaluation codes are released.
- Aug 28, 2024: 🤗 LLaVA-MoD is featured on [Huggingface Daily Papers](https://huggingface.co/papers/2408.15881).
- Aug 28, 2024: 📖 Paper is available on [Arxiv](https://arxiv.org/pdf/2408.15881). 

🌟 **Star us if you think it's helpful.** Your support means a lot! ⭐️

---

## ✨ Contents

- [🧭 Overview](#-overview)
- [🛠️ Installation](#-installation)
- [🗂️ Data Construction](#-data-construction)
- [🏋️‍♂️ Training and Evaluation](#-training-and-evaluation)
- [🚀 Inference](#-inference)
- [📖 Citation](#-citation)
- [🏆 Acknowledgement](#-acknowledgement)
- [📄 License](#-license)

---

## 🧭 Overview

**<b>TL; DR:</b>** LLaVA-MoD is an efficient framework for training small-scale Multimodal Language Models by distilling knowledge from larger models.

<details>
  <summary>🚀 <b>CLICK for the full abstract</b></summary>

We introduce **LLaVA-MoD**, a novel framework designed to enable the efficient training of small-scale Multimodal Language Models by distilling knowledge from large-scale MLLM. Our approach addresses two fundamental challenges in MLLM distillation:

- **Network Optimization**: We enhance the s-MLLM structure by integrating a sparse Mixture of Experts (MoE) architecture, balancing computational efficiency and model expressiveness.
  
- **Progressive Knowledge Transfer**: We propose a two-stage transfer strategy:
  1. **Mimic Distillation**: Minimizing Kullback-Leibler (KL) divergence between output distributions to help the student model emulate the teacher's understanding.
  2. **Preference Distillation**: Using Direct Preference Optimization (DPO), where the student model learns to outperform the teacher, especially in hallucination benchmarks.

Extensive experiments show **LLaVA-MoD** outperforms existing models across multimodal benchmarks while activating only a minimal number of parameters and keeping computational costs low. With **only 2B activated parameters**, **LLaVA-MoD** surpasses **Qwen-VL-Chat-7B** by an average of **8.8%**, using merely **0.3% of the training data** and **23% trainable parameters**.

These results highlight **LLaVA-MoD**’s success in distilling comprehensive knowledge from its teacher model, making it a groundbreaking solution for developing more efficient MLLMs.
</details>

---

## 🛠️ Installation

1. First install `anaconda`, and install `torch`, We recommend installing `torch==2.1.2` and `cuda==11.8`.

```bash
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

2. Then install the packages in `requirements`
```bash
pip install -r requirements.txt
```
---

## 🗂️ Data Construction
### 📚 Mimic Distillation
We follow LLaVA to construct the data as following format:
```json
{
  "id": "000000052846",
  "image": "COCO2017/train/000000052846.jpg",
  "conversations": [
     {
        "from": "human",
        "value": "Where is the cat positioned in the image?\n<image>"
     },
     {
        "from": "gpt",
        "value": "The cat is positioned on top of the back of the couch in the living room."
     },
     {
        "from": "human",
        "value": "What is the cat doing in the image?"
     },
     {
        "from": "gpt",
        "value": "The cat is coming out from some curtains onto the couch and is sitting or standing on top of it."
     }
  ]
}
```

### 🧠 Preference Distillation
We follow RLAIF-V to construct the data as following format:
```json
{
    "id": "OmniLMM-12B_OmniLMM-12B_0",
    "image": "RLAIF-V-Dataset_000_OmniLMM-12B_OmniLMM-12B_0.jpg",
    "chosen": [
      {
        "from": "human",
        "value": "<image>\nWho is more likely to use these tools a leather crafter or a paper crafter?"
      },
      {
        "from": "gpt",
        "value": "A leather crafter is more likely to use these tools. The image shows various crafting tools, including scissors and a hole punch, which are commonly used in leatherworking projects. Leather is a material that requires cutting, shaping, and precise hole-punching techniques to create desired designs or patterns. In contrast, paper crafters typically use different types of tools, such as adhesives, decorative papers, or specialized cutting machines like the Silhouette Cameo, for their projects."
      }
    ],
    "rejected": [
      {
        "from": "human",
        "value": "<image>\nWho is more likely to use these tools a leather crafter or a paper crafter?"
      },
      {
        "from": "gpt",
        "value": "A leather crafter is more likely to use these tools as they consist of a hole punch, scissors, and a knife. These items are typically used in crafting projects involving fabric or leather materials for various designs and patterns. Paper crafters may also benefit from some of these tools, but their primary focus would be on paper-related projects, which might require different types of tools such as paper cutters or scrapbooking supplies."
      }
    ]
}
```

---

## 🏋️‍♂️ Training and Evaluation
The full details for training and evaluation can be found in the [TRAIN_EVAL.md](docs/TRAIN_EVAL.md).

---

## 🚀 Inference
For instructions on inference, please refer to the [INFERENCE.md](docs/INFERENCE.md).

---

## 📖 Citation
If you find our project useful for your research and applications, please star it and cite the paper using this BibTeX:
```BibTeX
@article{shu2024llavamod,
  title={LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation},
  author={Shu, Fangxun and Liao, Yue and Zhuo, Le and Xu, Chenning and Zhang, Lei and Zhang, Guanghao and Shi, Haonan and Chen, Long and Zhong, Tao and He, Wanggui and Fu, Siming and others},
  journal={arXiv preprint arXiv:2408.15881},
  year={2024}
}
```
---

## 🏆 Acknowledgement
Our project is built upon [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA) and [LLaVA](https://github.com/haotian-liu/LLaVA). We are deeply grateful for the excellent codebase they provide. Additionally, we express our appreciation to [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM) and [RLAIF-V](https://github.com/RLHF-V/RLAIF-V) for their meticulously processed datasets. Their contributions have been of immeasurable value in shaping our work.

---

## 📄 License
Our project is released under the Apache 2.0 license.

