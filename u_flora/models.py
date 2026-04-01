import math

import torch
from omegaconf import DictConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.name,
        torch_dtype=torch.float32,
    )

    # model = prepare_model_for_kbit_training(
    #     model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    # )

    lora_target_modules = model_cfg.lora.peft_target_modules.split(",")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=model_cfg.lora.peft_lora_dropout,
        target_modules=lora_target_modules,
    )

    return get_peft_model(model, peft_config)
