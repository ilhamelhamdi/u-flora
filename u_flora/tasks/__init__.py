"""Abstract base for task-specific adapters in federated LoRA fine-tuning.

Concrete adapters live in sibling modules:
  - ``text_classification.py``  (SEQ_CLS: SST-2, QNLI, QQP, ...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from omegaconf import DictConfig
from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class TaskAdapter(ABC):
    """One adapter per NLP task type.

    Lifecycle:
      1. ``get_model()``          — load pretrained + wrap with LoRA
      2. ``get_tokenizer()``      — load tokenizer
      3. ``get_encoding_fn()``    — dataset .map() callable
      4. ``get_data_collator()``  — batching / padding strategy
      5. ``compute_metrics()``    — eval metric computation
    """

    # ---- Model ---------------------------------------------------------------

    @abstractmethod
    def get_model(self, model_cfg: DictConfig) -> PreTrainedModel:
        """Return a PEFT-wrapped model ready for LoRA fine-tuning."""
        ...

    @abstractmethod
    def get_peft_task_type(self) -> str:
        """Return the PEFT ``TaskType`` string for this task.

        Examples: ``"SEQ_CLS"``.
        """
        ...

    @abstractmethod
    def get_model_size_kb(self, model_cfg: DictConfig) -> int:
        """Return the size of the pretrained model in KB (for reporting upload/download costs)."""
        ...

    # ---- Tokenization --------------------------------------------------------

    @abstractmethod
    def get_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        """Load the tokenizer for the given model."""
        ...

    @abstractmethod
    def get_encoding_fn(
        self,
        model_name: str,
        dataset_cfg: DictConfig,
    ) -> Callable:
        """Return a function suitable for ``dataset.map(fn, batched=True)``."""
        ...

    @abstractmethod
    def get_data_collator(self, model_name: str) -> Any:
        """Return a data collator for the Trainer."""
        ...

    # ---- Evaluation ----------------------------------------------------------
    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the main evaluation metric for this task."""
        ...

    @abstractmethod
    def is_higher_metric_better(self) -> bool:
        """Return whether higher metric values are better (e.g. accuracy vs perplexity)."""
        ...

    @abstractmethod
    def compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        """Compute task-specific metrics from ``(predictions, labels)``."""
        ...

    # ---- Helpers -------------------------------------------------------------

    def build_lora_config(self, lora_cfg: DictConfig) -> LoraConfig:
        """Build a ``LoraConfig`` from the Hydra/OmegaConf sub-config.

        Default implementation reads standard fields; override if a task
        needs non-standard LoRA settings.
        """
        target_modules = lora_cfg.peft_target_modules
        if isinstance(target_modules, str):
            target_modules = [m.strip() for m in target_modules.split(",")]

        return LoraConfig(
            task_type=self.get_peft_task_type(),
            r=int(lora_cfg.peft_lora_r),
            lora_alpha=int(lora_cfg.peft_lora_alpha),
            lora_dropout=float(lora_cfg.peft_lora_dropout),
            target_modules=target_modules,
        )
