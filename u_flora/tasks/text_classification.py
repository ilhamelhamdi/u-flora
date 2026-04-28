"""Text classification adapter for GLUE-style tasks.

Supports single-sentence (SST-2) and sentence-pair (QNLI, QQP) tasks.
Uses ``AutoModelForSequenceClassification`` with LoRA ``TaskType.SEQ_CLS``.
"""

from __future__ import annotations

from typing import Any, Callable

import evaluate
import numpy as np
import math
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

from . import TaskAdapter


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
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        lora_config = self.build_lora_config(model_cfg.lora)
        return get_peft_model(model, lora_config)

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

    def get_data_collator(self, model_name: str) -> Any:
        tokenizer = self.get_tokenizer(model_name)
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
