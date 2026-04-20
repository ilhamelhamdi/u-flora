"""Dataset loading and partitioning for federated learning.

Supports both IID and non-IID (Dirichlet) partitioning.
The FederatedDataset is cached globally per-process.
"""

from __future__ import annotations

import logging

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Global cache — one FederatedDataset per process
_FDS_CACHE: FederatedDataset | None = None
_FDS_KEY: str | None = None


def _get_partitioner(num_partitions: int, partition_config: DictConfig):
    """Create a partitioner based on the strategy name.

    Args:
        num_partitions: Number of FL client partitions.
        partition_config: Config with partitioning parameters: ``strategy``, ``dirichlet.alpha``, ``dirichlet.seed``.
    """
    strategy = partition_config.strategy
    if strategy == "dirichlet":
        try:
            from flwr_datasets.partitioner import DirichletPartitioner

            return DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=partition_config.dirichlet.alpha,
                seed=partition_config.dirichlet.seed,
            )
        except ImportError:
            logger.warning("DirichletPartitioner not available, falling back to IID")
    return IidPartitioner(num_partitions=num_partitions)


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_config: DictConfig,
    partition_config: DictConfig,
):
    """Load a client's data partition.

    Args:
        partition_id: This client's partition index.
        num_partitions: Total number of partitions.
        dataset_config: Config with ``name``, ``subset`` fields.
        partition_config: Config with partitioning parameters: ``strategy``, ``dirichlet.alpha``, ``dirichlet.seed``.

    Returns:
        (train_set, val_set) tuple of HuggingFace datasets.
    """
    global _FDS_CACHE, _FDS_KEY

    # Build a cache key to detect config changes
    subset = getattr(dataset_config, "subset", None)
    cache_key = (
        f"{dataset_config.name}:{subset}:{num_partitions}:{partition_config.strategy}"
    )

    if _FDS_CACHE is None or _FDS_KEY != cache_key:
        train_partitioner = _get_partitioner(num_partitions, partition_config)
        val_partitioner = IidPartitioner(num_partitions=num_partitions)

        _FDS_CACHE = FederatedDataset(
            dataset=dataset_config.name,
            subset=subset,
            partitioners={
                "train": train_partitioner,
                "validation": val_partitioner,
            },
        )
        _FDS_KEY = cache_key
        logger.info(
            "Initialized FederatedDataset: %s/%s, %d partitions (%s)",
            dataset_config.name,
            subset,
            num_partitions,
            partition_config.strategy,
        )

    train_set = _FDS_CACHE.load_partition(partition_id, "train")
    val_set = _FDS_CACHE.load_partition(partition_id, "validation")
    return train_set, val_set


def load_data_centralized(dataset_config: DictConfig, split: str = "validation"):
    """Load the full (non-partitioned) dataset for centralized evaluation.

    Used by server_app.py for global evaluation.
    """
    from datasets import load_dataset

    subset = getattr(dataset_config, "subset", None)
    return load_dataset(dataset_config.name, subset, split=split)
