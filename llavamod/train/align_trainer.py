import os
import warnings
from copy import deepcopy
from collections import defaultdict
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Union
)

from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from datasets import Dataset

from accelerate.utils import is_deepspeed_available

if is_deepspeed_available():
    import deepspeed

from llavamod.model.utils import create_reference_model, disable_dropout_in_model

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in
                    get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: Optional[List[int]] = None,
            generator=None,
            group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                          generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                 generator=self.generator)
        return iter(indices)


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class AlignTrainer(Trainer):
    r"""
    Initialize AlignTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`transformers.PreTrainedModel`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, default to 0):
            The robust DPO label smoothing parameter that should be between 0 and 0.5.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss,`"hinge"` loss from SLiC paper or `"ipo"` from IPO paper.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        disable_dropout (`bool`, defaults to `True`):
            Whether to disable dropouts in `model` and `ref_model`.
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module, str] = None,
            ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            label_pad_token_id: int = -100,
            padding_value: int = 0,
            beta: float = 0.1,
            label_smoothing: float = 0,
            loss_type: str = "sigmoid",
            moe_loss_enable: bool = False,
            disable_dropout: bool = True,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):

        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        if getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if ref_model:
            self.ref_model = ref_model
        else:
            self.ref_model = create_reference_model(model)

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self.moe_loss_enable = moe_loss_enable

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if not hasattr(self.accelerator.unwrap_model(self.model), "disable_adapter"):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your "
                    "`peft` version to the latest version."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.ref_model.module.eval()
        # self.ref_model.eval()

        print("##### ref model exists ", self.ref_model is not None)
        print("##### ref_model is in training mode:", self.ref_model.module.training)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Set up the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
                ]
            # import ipdb
            # ipdb.set_trace()
            # print('###### self.args', self.args)
            # print('###### self.args.moe_enable', self.args.moe_enable)

            if self.args.moe_enable:
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(
                    optimizer_grouped_parameters)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        # print("####### optimizer", self.optimizer)

        return self.optimizer

    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        # 设置 ZeRO Stage 2 优化
        config_kwargs["zero_optimization"]["stage"] = 2
        # print("#### deepspeed config for ref_model")
        # print(config_kwargs)

        # 通过 DeepSpeed 初始化模型
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        dummty_optimizer = DeepSpeedCPUAdam(model.parameters(), lr=0.0001)
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs, optimizer=dummty_optimizer,
                                         model_parameters=None)
        model.eval()  # FIXME: is it valid?
        # model.module.eval()

        return model

    def get_p(self, model, inputs):
        """
        """
        outputs = model(**inputs, return_dict=True)

        logits = outputs.logits
        labels = outputs.labels

        sft_loss = outputs.loss

        if self.args.moe_enable and self.moe_loss_enable and hasattr(outputs, 'moe_loss'):
            moe_loss = outputs.moe_loss
        else:
            moe_loss = None

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        logits_aligned = logits[:, :, :151936]

        probs = F.softmax(logits_aligned, dim=-1, dtype=torch.float32)  # probs

        return probs, sft_loss, moe_loss

    def get_logp(self, model, inputs):
        """
        """
        outputs = model(**inputs, return_dict=True)

        logits = outputs.logits
        labels = outputs.labels

        sft_loss = outputs.loss

        if self.args.moe_enable and self.moe_loss_enable and hasattr(outputs, 'moe_loss'):
            moe_loss = outputs.moe_loss
        else:
            moe_loss = None

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        logits_aligned = logits[:, :, :151936]  # NOTE: FIXED ME

        logprobs = F.log_softmax(logits_aligned, dim=-1, dtype=torch.float32)  # student logp

        return logprobs, sft_loss, moe_loss, labels

    def compute_align_loss(self, policy_logprobs, reference_probs, labels):

        # print("#### policy_logprobs: ", policy_logprobs.size())
        # print("#### reference_probs: ", reference_probs.size())
        # print("#### labels: ", labels.size())

        inf_mask = torch.isinf(policy_logprobs)
        prod_probs = torch.masked_fill(reference_probs * policy_logprobs, inf_mask, 0)

        # print("#### prod_probs: ", prod_probs.size())

        x = torch.sum(prod_probs, dim=-1).view(-1)

        # print("#### x: ", x.size())
        if self.args.distill_all_tokens:
            # If we're using all tokens, the loss mask will be all ones
            print("####### compute multimodal instruction + response tokens")
            loss_mask = torch.ones_like(labels).int()
        else:
            loss_mask = (labels != self.label_pad_token_id).int()  # modify it to distill text instruction + vision tokens

        # print("#### loss_mask: ", loss_mask.size())

        align_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)

        return align_loss

    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        """
        # print('#### chosen_input_ids: ', inputs['chosen_input_ids'].device)
        # print('#### chosen_labels: ', inputs['chosen_labels'].device)
        # print('#### chosen_attention_mask: ', inputs['chosen_attention_mask'].device)
        # print('#### rejected_input_ids: ', inputs['rejected_input_ids'].device)
        # print('#### rejected_labels: ', inputs['rejected_labels'].device)
        # print('#### rejected_attention_mask: ', inputs['rejected_attention_mask'].device)

        # inputs = dict(
        #     input_ids=inputs['input_ids'],
        #     labels=inputs['labels'],
        #     attention_mask=inputs['attention_mask'],
        # )

        if 'images' in inputs:
            inputs['images'] = inputs['images']

        assert self.ref_model is not None, "ref model can not be none!"

        with torch.no_grad():
            if self.ref_model is None:
                raise
            else:
                reference_probs, *_ = self.get_p(self.ref_model, inputs)

        policy_logprobs, policy_sft_loss, policy_moe_loss, policy_labels = self.get_logp(model, inputs)



        align_loss = self.compute_align_loss(policy_logprobs=policy_logprobs,
                                             reference_probs=reference_probs,
                                             labels=policy_labels)

        if self.loss_type == "only_kd":
            losses = align_loss
        else:
            losses = align_loss + policy_sft_loss

        if policy_moe_loss:
            moe_loss = policy_moe_loss
            losses = losses + moe_loss
        else:
            moe_loss = torch.full_like(align_loss, -1.0)
            losses = losses

        outputs = {
            "loss": losses.mean(),
            "loss/align": align_loss.mean(),
            "loss/moe_balance": moe_loss.mean() if not isinstance(moe_loss, int) else moe_loss,
            "loss/lm": policy_sft_loss.mean()
        }

        self.store_metrics(outputs, train_eval="train")

        if return_outputs:
            return losses.mean(), outputs
        else:
            return losses.mean()

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either have 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            print("######## saving checkpoint........")
            super(AlignTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            # print("######## saving checkpoint........")
            super(AlignTrainer, self)._save(output_dir, state_dict)
