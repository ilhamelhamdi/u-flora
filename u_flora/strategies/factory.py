import logging
from omegaconf import DictConfig

from . import BaseStrategy, RandomStrategy, FedCSStrategy
from ..client_profile.typing import ClientState

logger = logging.getLogger(__name__)

def build_strategy(
    cfg: DictConfig,
    client_states: dict[int, ClientState],
    save_path: str,
    use_wandb: bool = True,
) -> BaseStrategy:
    """Instantiate the strategy-specific class.

    Reads from ``cfg.strategy`` (name: ``random`` | ``fedcs``).
    """
    strategy_cfg = cfg.strategy
    name = str(getattr(strategy_cfg, "name", "random")).lower()
    num_to_select = int(getattr(strategy_cfg, "num_to_select", 10))
    seed = int(getattr(strategy_cfg, "seed", 42))

    common = dict(
        client_states=client_states,
        save_path=save_path,
        use_wandb=use_wandb,
    )

    if name == "random":
        return RandomStrategy(num_to_select=num_to_select, seed=seed, **common)

    if name == "fedcs":
        local_epochs = int(
            getattr(
                getattr(cfg.train, "training_arguments", None),
                "num_train_epochs",
                1,
            )
        )
        return FedCSStrategy(
            num_to_select=num_to_select,
            round_deadline_s=float(getattr(strategy_cfg, "round_deadline_s", 300.0)),
            model_size_kb=float(getattr(strategy_cfg, "model_size_kb", 1000.0)),
            local_epochs=local_epochs,
            c_fraction=float(getattr(strategy_cfg, "c_fraction", 0.5)),
            exploration_fraction=float(
                getattr(strategy_cfg, "exploration_fraction", 0.1)
            ),
            seed=seed,
            **common,
        )

    logger.warning("Unknown strategy '%s', falling back to Random", name)
    return RandomStrategy(num_to_select=num_to_select, seed=seed, **common)
