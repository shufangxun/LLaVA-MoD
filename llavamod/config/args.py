from typing import Optional, List
from dataclasses import field

from llavamod.data.dataset import *
from llavamod.train.train_utils import *


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_llm_ffn_only: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    s2: bool = field(default=False)
    s2_scales: Optional[str] = field(default="336,672")

    # ===================================================================
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    image_projector_type: Optional[str] = field(default='linear')
    video_projector_type: Optional[str] = field(default='linear')
    video_global_proj: bool = field(default=False)
    video_temproal_proj: bool = field(default=False)
    video_spatial_proj: bool = field(default=False)
    # ===================================================================

    # =============================================================
    only_lora_ffn: bool = True
    moe_enable: bool = False
    train_modules: Optional[List[str]] = field(default=None, metadata={"help": ""})
    moe_mode: str = field(
        default="second_half",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["first_half", "second_half", "sparse", "dense"],
        },
    )
    moe_layers_idx: Optional[List[int]] = field(default=None, metadata={"help": "where to place moe layers."})
    ep_size: int = 1
    num_experts: Optional[List[int]] = field(default=4, metadata={"help": "number of experts for each moe layer."})
    top_k_experts: int = field(
        default=2,
        metadata={
            "help": "Top-k experts to deal with tokens.",
            "choices": [1, 2, 3, 4],
        },
    )
    capacity_factor: float = 1.
    eval_capacity_factor: float = 2.
    min_capacity: int = 0
    use_residual: bool = False
    router_aux_loss_coef: float = 0.01
    # =============================================================


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    # ===================================================================
    data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: int = 8
    # ===================================================================


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    moe_finetune: bool = field(default=False)
    distill_all_tokens: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})


@dataclass
class AlignArguments:
    policy_model_type: str = field(default='sparse')
    ref_model_type: str = field(default='dense')
    loss_type: str = field(default='only_kd')
    policy_model_name_or_path: str = field(default=None)
    policy_pretrain_mm_mlp_adapter: str = field(default=None)
    ref_model_name_or_path: str = field(default=None)
    ref_pretrain_mm_mlp_adapter: str = field(default=None)
    moe_loss_enable: bool = field(default=False)


@dataclass
class DPOArguments:
    policy_model_type: str = field(default='sparse')
    ref_model_type: str = field(default='dense')
    loss_type: str = field(default='sigmoid')
    policy_model_name_or_path: str = field(default=None)
    ref_model_name_or_path: str = field(default=None)
    moe_loss_enable: bool = field(default=False)


