"""u_flora.strategies — per-strategy federated learning classes.

Each strategy class combines client selection logic and Flower integration
into a single cohesive unit.  All strategies extend ``BaseStrategy``, which
handles node mapping, FedAvg aggregation, and comprehensive W&B metrics.

Usage
-----
    from u_flora.strategies import RandomStrategy, FedCSStrategy, OortStrategy

    strategy = RandomStrategy(
        num_to_select=10,
        client_states=client_states,
        save_path=save_path,
    )
    result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=50)
"""

from .base import BaseStrategy
from .random import RandomStrategy
from .fedcs import FedCSStrategy


__all__ = [
    "BaseStrategy",
    "RandomStrategy",
    "FedCSStrategy",
]
