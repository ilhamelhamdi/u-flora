"""Shared utility functions."""

from __future__ import annotations

import logging
import math

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    pkg_logger = logging.getLogger("u_flora")
    if pkg_logger.handlers:
        return  # already configured in this process

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    pkg_logger.propagate = False  # don't double-emit if root is later configured


def replace_keys(
    input_dict: dict, match: str = "-", target: str = "_"
) -> dict:
    """Recursively replace ``match`` with ``target`` in dictionary keys.

    Flower configs use hyphens in keys (e.g. ``learning-rate-max``)
    but OmegaConf / Python attrs use underscores.
    """
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Cosine annealing learning rate schedule.

    Decays learning rate from ``lrate_max`` to ``lrate_min`` over
    ``total_round`` rounds following a cosine curve.
    """
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))
