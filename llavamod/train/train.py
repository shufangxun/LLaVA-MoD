import wandb
import pathlib
from glob import glob

from llavamod.config.args import ModelArguments, DataArguments, TrainingArguments
from llavamod.data.dataset import *
from llavamod.model import *
from llavamod.train.train_utils import *
from llavamod.train.llava_trainer import LLaVATrainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train():
    global local_rank

    # ==================== global config =======================
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    # ==================== model config =======================
    if model_args.image_tower is not None or model_args.video_tower is not None:
        if not model_args.moe_enable:
            if 'mpt' in model_args.model_name_or_path.lower():
                config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                config.attn_config['attn_impl'] = training_args.mpt_attn_impl
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    **bnb_model_from_pretrained_args
                )
            elif 'qwen' in model_args.model_name_or_path.lower():
                if 'qwen2' in model_args.model_name_or_path.lower() or 'qwen-2' in model_args.model_name_or_path.lower():
                    print("####### using qwen2 dense........")
                    model = LlavaQwen2ForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        attn_implementation=training_args.attn_implementation,
                        # torch_dtype=torch.bfloat16,
                        **bnb_model_from_pretrained_args
                    )
                elif 'qwen1.5' in model_args.model_name_or_path.lower() or 'qwen-1.5' in model_args.model_name_or_path.lower():
                    model = LlavaQwen1_5ForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        attn_implementation=training_args.attn_implementation,
                        # torch_dtype=torch.bfloat16,
                        **bnb_model_from_pretrained_args
                    )
                else:
                    model = LlavaQWenForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        attn_implementation=training_args.attn_implementation,
                        **bnb_model_from_pretrained_args
                    )
            elif "gemma-2" in model_args.model_name_or_path.lower():
                model = LlavaGemma2ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation="eager",
                    # torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            elif 'openchat' in model_args.model_name_or_path.lower() or 'mistral' in model_args.model_name_or_path.lower():
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'phi' in model_args.model_name_or_path.lower():
                model = LlavaPhiForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'minicpm' in model_args.model_name_or_path.lower():
                model = LlavaMiniCPMForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'stablelm' in model_args.model_name_or_path.lower():
                model = LlavaStablelmForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
        else:
            if 'qwen' in model_args.model_name_or_path.lower():
                if 'qwen2' in model_args.model_name_or_path.lower() or 'qwen-2' in model_args.model_name_or_path.lower():
                    print("####### using qwen2 moe........")
                    if not training_args.moe_finetune:
                        model = LLaVAMoDQwen2ForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            attn_implementation=training_args.attn_implementation,
                            # torch_dtype=torch.bfloat16,
                            **bnb_model_from_pretrained_args
                        )
                    else:
                        model = LLaVAMoDQwen2ForCausalLMFineTune.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            attn_implementation=training_args.attn_implementation,
                            # torch_dtype=torch.bfloat16,
                            **bnb_model_from_pretrained_args
                        )
                elif 'qwen1.5' in model_args.model_name_or_path.lower() or 'qwen-1.5' in model_args.model_name_or_path.lower():
                    if not training_args.moe_finetune:
                        model = LLaVAMoDQwen1_5ForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            attn_implementation=training_args.attn_implementation,
                            # torch_dtype=torch.bfloat16,
                            **bnb_model_from_pretrained_args
                        )
                    else:
                        model = LLaVAMoDQwen1_5ForCausalLMFineTune.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            attn_implementation=training_args.attn_implementation,
                            # torch_dtype=torch.bfloat16,
                            **bnb_model_from_pretrained_args
                        )
                else:
                    if not training_args.moe_finetune:
                        model = LLaVAMoDQWenForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            **bnb_model_from_pretrained_args
                        )
                    else:
                        model = LLaVAMoDQWenForCausalLMFineTune.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            attn_implementation=training_args.attn_implementation,
                            # torch_dtype=torch.bfloat16,
                            **bnb_model_from_pretrained_args
                        )
            elif "gemma-2" in model_args.model_name_or_path.lower():
                if not training_args.moe_finetune:
                    model = LLaVAMoDGemma2ForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        attn_implementation="eager",
                        # torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                        **bnb_model_from_pretrained_args
                    )
                else:
                    model = LLaVAMoDGemma2ForCausalLMFineTune.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        attn_implementation=training_args.attn_implementation,
                        # torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                        **bnb_model_from_pretrained_args
                    )
            elif 'phi' in model_args.model_name_or_path.lower():
                model = LLaVAMoDPhiForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'minicpm' in model_args.model_name_or_path.lower():
                model = LLaVAMoDMiniCPMForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'openchat' in model_args.model_name_or_path.lower() or 'mistral' in model_args.model_name_or_path.lower():
                model = LLaVAMoDMistralForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'stablelm' in model_args.model_name_or_path.lower():
                model = LLaVAMoDStablelmForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            else:
                model = LLaVAMoDLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            # torch_dtype=torch.bfloat16,
            **bnb_model_from_pretrained_args
        )

    # rank0_print('LLM init. firstly\n', model)
    model.config.use_cache = False

    """
    冻结backbone
    """
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if model_args.tune_llm_ffn_only:
        train_modules = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        for name, param in model.named_parameters():
            if "image_tower" not in name:
                if any(n in name for n in train_modules):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                continue

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ==================== moe config =======================
    training_args.moe_enable = model_args.moe_enable
    model_args.lora_enable = training_args.lora_enable
    training_args.only_lora_ffn = model_args.only_lora_ffn
    if model_args.moe_enable:
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            if 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
                target_modules = [
                    'mlp.w1', 'mlp.w2', 'mlp.c_proj'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            elif 'phi' in model_args.model_name_or_path.lower():
                target_modules = [
                    'fc1', 'fc2'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            else:
                target_modules = [
                    'up_proj', 'down_proj', 'gate_proj'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            # modules_to_save = ['wg']  # weight gating for MoE
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                # modules_to_save=modules_to_save,
                task_type="CAUSAL_LM",
            )
            model_args.lora_r = training_args.lora_r
            model_args.lora_alpha = training_args.lora_alpha
            model_args.lora_dropout = training_args.lora_dropout
            model_args.lora_bias = training_args.lora_bias
            # model_args.modules_to_save = modules_to_save
            model_args.target_modules = target_modules
            model_args.train_modules = target_modules
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)

        # if not training_args.moe_finetune:
        #     model.initialize_moe_modules(model_args=model_args)
        model.initialize_moe_modules(model_args=model_args)

    else:
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)

    # ======================= tokenzier ===========================
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        # import ipdb
        # ipdb.set_trace()
        if 'qwen' in model_args.model_name_or_path.lower():
            if 'qwen2' in model_args.model_name_or_path.lower() or 'qwen-2' in model_args.model_name_or_path.lower() or 'qwen1.5' in model_args.model_name_or_path.lower() or 'qwen-1.5' in model_args.model_name_or_path.lower():  # qwen 1.5 + 2.0
                print("#### using qwen2 or qwen1.5 tokenizer.....")
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    model_max_length=training_args.model_max_length,
                    padding_side="right",
                    use_fast=False,
                )
                tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
            else:  # qwen 1.0
                from llavamod.model.language_model.qwen.tokenization_qwen import QWenTokenizer
                tokenizer = QWenTokenizer.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    model_max_length=training_args.model_max_length,
                    padding_side="right",
                    use_fast=False,
                )
                tokenizer.add_special_tokens({'unk_token': '<|extra_0|>', 'eos_token': '<|endoftext|>'})
                # print("################# Qwen !!!")
                # print(tokenizer.unk_token)
        elif "gemma-2" in model_args.model_name_or_path.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
        elif 'phi' in model_args.model_name_or_path.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
        elif 'stablelm' in model_args.model_name_or_path.lower():
            from llavamod.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
            tokenizer = Arcade100kTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.unk_token = '<|reg0|>'  # FIXME: DO SUPPORT ADD SPECIAL TOKENS
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )

    # print("############ Qwen !!!!!! ########")
    # print(tokenizer.unk_token)
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "gemma_2":
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["gemma_2"]
    else:
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # print(conversation_lib.default_conversation)
    # =============================================================================================================
    if model_args.image_tower is not None or model_args.video_tower is not None:
        # print(model_args)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        if model_args.image_tower is not None:
            image_tower = model.get_image_tower()
            image_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.image_processor = image_tower.image_processor
            data_args.is_multimodal = True
        if model_args.video_tower is not None:
            video_tower = model.get_video_tower()
            video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.video_processor = video_tower.video_processor
            data_args.is_multimodal = True
            data_args.num_frames = video_tower.config.num_frames
        # =============================================================================================================

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side

        model.config.s2 = model_args.s2
        model.config.s2_scales = model_args.s2_scales

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    rank0_print('####### Vision encoder and proj init !!!!!!!\n', model)
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    rank0_print("##### trainable model parameters!!!!")
    for name, param in model.named_parameters():
        # if param.requires_grad:
        rank0_print(name, param.requires_grad)

    # import pdb
    # pdb.set_trace()

    rank0_print("##### making dataset!!!!")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable and not model_args.moe_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        if model_args.moe_enable:
            ckpt = model.state_dict()
            ckpt = {(k[11:] if k.startswith('base_model.') else k): v for k, v in ckpt.items()}
            if any(k.startswith('model.model.') for k in ckpt):
                ckpt = {(k[6:] if k.startswith('model.') else k): v for k, v in ckpt.items()}
            torch.save(ckpt, os.path.join(training_args.output_dir, 'pytorch_model.bin'))
            model.config.save_pretrained(training_args.output_dir)
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                [os.remove(i) for i in glob(os.path.join(training_args.output_dir, 'adapter_*'))]
    # print(model.state_dict().keys())


if __name__ == "__main__":
    train()
