"""RandomStrategy: uniform random client selection.

Each round, ``num_to_select`` clients are sampled uniformly at random from
the full pool of registered clients.  No pre-training phase, no per-round
resource requests — just random sampling followed by standard FedAvg.
"""

from __future__ import annotations

import random
from typing import Any

from flwr.common import ArrayRecord, Message
from flwr.server import Grid

from .base import BaseStrategy
from ..selection.base import ClientState


class RandomStrategy(BaseStrategy):
    """Uniform random client selection.

    Args:
        num_to_select: Number of clients to select each round (K).
        seed: Random seed for reproducibility.
        **kwargs: Forwarded to BaseStrategy (client_states, save_path, use_wandb).
    """

    def __init__(
        self,
        num_to_select: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_to_select = num_to_select
        self._rng = random.Random(seed)

    def configure_train(
        self,
        round_num: int,
        arrays: ArrayRecord,
        grid: Grid,
        timeout: float,
    ) -> tuple[list[int], list[Message]]:
        all_pids = list(self.client_states.keys())
        k = min(self.num_to_select, len(all_pids))
        selected = self._rng.sample(all_pids, k)
        messages = self._make_train_messages(selected, arrays, round_num)
        return selected, messages

    def _strategy_name(self) -> str:
        return f"Random(K={self.num_to_select})"
