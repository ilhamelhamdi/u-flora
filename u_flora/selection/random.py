"""Random client selection — the simplest baseline.

Selects K clients uniformly at random each round.
This is the default strategy used by FedAvg (McMahan et al., 2017).
"""

from __future__ import annotations

import random
from typing import Any

from .base import ClientSelector, ClientState


class RandomSelector(ClientSelector):
    """Uniform random client selection.

    This is the baseline used in most FL papers. It makes no use of
    statistical or system utility — purely random each round.
    """

    def __init__(
        self,
        num_to_select: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_to_select, **kwargs)
        self._rng = random.Random(seed)

    def select_clients(
        self,
        current_round: int,
        client_states: dict[int, ClientState],
    ) -> list[int]:
        candidates = list(client_states.keys())
        k = min(self.num_to_select, len(candidates))
        return self._rng.sample(candidates, k)

    @property
    def name(self) -> str:
        return "Random"