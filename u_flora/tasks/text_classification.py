"""Text classification adapter for GLUE-style tasks.

Supports single-sentence (SST-2) and sentence-pair (QNLI, QQP) tasks.
Uses ``AutoModelForSequenceClassification`` with LoRA ``TaskType.SEQ_CLS``.
"""

from __future__ import annotations

from typing import Any, Callable

import evaluate
import logging
import math
import numpy as np
import torch
from omegaconf import DictConfig
from peft import TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    from transformers.utils import is_flash_attn_2_available
except ImportError:  # pragma: no cover - older Transformers versions
    is_flash_attn_2_available = None

try:
    from transformers import DataCollatorWithFlattening
except ImportError:  # pragma: no cover - older Transformers versions
    DataCollatorWithFlattening = None

from . import TaskAdapter

logger = logging.getLogger(__name__)


class TextClassificationAdapter(TaskAdapter):
    """Adapter for sequence-classification tasks (GLUE family).

    Dataset config must include:
      - ``text_fields``: comma-separated field names
        (e.g. ``"sentence"`` for SST-2, ``"question1,question2"`` for QQP)
    """

    metric_name = "accuracy"
    _accuracy = evaluate.load("accuracy")

    # ---- Model ---------------------------------------------------------------

    def get_peft_task_type(self) -> str:
        return TaskType.SEQ_CLS

    def get_model(self, model_cfg: DictConfig) -> PreTrainedModel:
        num_labels = int(getattr(model_cfg, "num_labels", 2))
        attn_implementation = self._resolve_attn_implementation(model_cfg)
        torch_dtype = self._resolve_torch_dtype(model_cfg)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            attn_implementation=attn_implementation or None,
        )
        lora_config = self.build_lora_config(model_cfg.lora)
        return get_peft_model(model, lora_config)

    def _resolve_attn_implementation(self, model_cfg: DictConfig) -> str | None:
        attn_impl = getattr(model_cfg, "attn_implementation", None)
        if attn_impl:
            if attn_impl == "flash_attention_2" and not self._flash_attn_available():
                logger.warning(
                    "Requested flash_attention_2 but flash_attn is not installed; "
                    "falling back to default attention."
                )
                return None
            return attn_impl
        return None

    def _flash_attn_available(self) -> bool:
        if is_flash_attn_2_available is None:
            return False
        return bool(is_flash_attn_2_available())

    def _resolve_torch_dtype(self, model_cfg: DictConfig) -> torch.dtype:
        raw_dtype = getattr(model_cfg, "torch_dtype", None)
        if raw_dtype is None:
            return torch.float32
        if isinstance(raw_dtype, torch.dtype):
            return raw_dtype
        if isinstance(raw_dtype, str):
            dtype_key = raw_dtype.strip().lower()
            if dtype_key == "auto":
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float32
            if dtype_key in {"bf16", "bfloat16"}:
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float32
            if dtype_key in {"fp16", "float16"}:
                return torch.float16
            if dtype_key in {"fp32", "float32"}:
                return torch.float32
        return torch.float32

    def get_lora_adapter_size_kb(self, model_cfg: DictConfig) -> int:
        peft_model = self.get_model(model_cfg)
        # Only count trainable parameters (the PEFT adapter)
        total_size_bytes = sum(
            p.numel() * p.element_size()
            for p in peft_model.parameters()
            if p.requires_grad
        )
        return math.ceil(total_size_bytes / 1024)

    # ---- Tokenization --------------------------------------------------------

    def get_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(model_name)

    def get_encoding_fn(
        self,
        model_name: str,
        dataset_cfg: DictConfig,
    ) -> Callable:
        tokenizer = self.get_tokenizer(model_name)
        text_fields = [f.strip() for f in dataset_cfg.text_fields.split(",")]

        def _encode(examples: dict) -> dict:
            inputs = [examples[field] for field in text_fields]
            return tokenizer(*inputs, truncation=True)

        return _encode

    def get_data_collator(
        self,
        model_cfg: DictConfig | None = None,
    ) -> Any:
        tokenizer = self.get_tokenizer(model_cfg.name)
        return DataCollatorWithPadding(tokenizer=tokenizer)

    # ---- Evaluation ----------------------------------------------------------

    def get_metric_name(self):
        return self.metric_name

    def is_higher_metric_better(self):
        return True

    def compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = self._accuracy.compute(predictions=predictions, references=labels)
        return {self.metric_name: acc["accuracy"]}
