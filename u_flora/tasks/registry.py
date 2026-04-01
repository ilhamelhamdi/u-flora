"""Task registry — resolves a task name string to its adapter instance.

Usage in client_app.py or server_app.py:
    from u_flora.tasks.registry import get_task_adapter
    adapter = get_task_adapter("text_classification")
"""

from __future__ import annotations

from . import TaskAdapter
from .text_classification import TextClassificationAdapter

_REGISTRY: dict[str, type[TaskAdapter]] = {
    "text_classification": TextClassificationAdapter,
    "seq_cls": TextClassificationAdapter,      # alias
}


def get_task_adapter(task_name: str) -> TaskAdapter:
    """Instantiate the adapter for the given task name.

    Raises:
        ValueError: If the task name is not registered.
    """
    cls = _REGISTRY.get(task_name.lower())
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task {task_name!r}. Available: {available}"
        )
    return cls()


def register_task(name: str, adapter_cls: type[TaskAdapter]) -> None:
    """Register a custom task adapter at runtime."""
    _REGISTRY[name.lower()] = adapter_cls
