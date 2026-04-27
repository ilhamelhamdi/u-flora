import logging

from omegaconf import DictConfig

from . import BaseStrategy, RandomStrategy, FedCSStrategy, TiFLStrategy, OortStrategy
from ..client_profile.typing import ClientState

logger = logging.getLogger(__name__)


def build_strategy(
    cfg: DictConfig,
    client_states: dict[int, ClientState],
    save_path: str,
    use_wandb: bool = True,
    metric_name: str = "accuracy",
    is_higher_better: bool = True,
) -> BaseStrategy:
    """Instantiate the strategy-specific class.

    Reads from ``cfg.strategy`` (name: ``random`` | ``fedcs`` | ``tifl`` | ``oort``).
    """
    strategy_cfg = cfg.strategy
    name = str(strategy_cfg.get("name", "random")).lower()
    num_to_select = strategy_cfg.num_to_select
    seed = strategy_cfg.get("seed", 42)
    local_epochs = cfg.train.training_arguments.num_train_epochs

    model_size_kb = float(cfg.get("model_size_kb", 0.0))
    heartbeat_timeout_s = float(cfg.get("heartbeat_timeout_s", 30.0))

    common = dict(
        client_states=client_states,
        save_path=save_path,
        use_wandb=use_wandb,
        metric_name=metric_name,
        is_higher_better=is_higher_better,
        model_size_kb=model_size_kb,
        heartbeat_timeout_s=heartbeat_timeout_s,
    )

    if name == "random":
        return RandomStrategy(num_to_select=num_to_select, seed=seed, **common)

    if name == "fedcs":
        return FedCSStrategy(
            round_deadline_s=strategy_cfg.round_deadline_s,
            local_epochs=local_epochs,
            c_fraction=strategy_cfg.c_fraction,
            seed=seed,
            **common,
        )

    if name == "tifl":
        return TiFLStrategy(
            num_to_select=num_to_select,
            num_tiers=strategy_cfg.num_tiers,
            sync_rounds=strategy_cfg.sync_rounds,
            prob_update_interval=strategy_cfg.prob_update_interval,
            round_deadline_s=strategy_cfg.round_deadline_s,
            local_epochs=local_epochs,
            seed=seed,
            **common,
        )

    if name == "oort":
        return OortStrategy(
            num_to_select=num_to_select,
            epsilon=strategy_cfg.epsilon,
            alpha=strategy_cfg.alpha,
            pacer_window=strategy_cfg.pacer_window,
            pacer_delta=strategy_cfg.pacer_delta,
            initial_t=strategy_cfg.initial_t,
            cutoff_c=strategy_cfg.cutoff_c,
            max_participation_rounds=strategy_cfg.max_participation_rounds,
            utility_clip_percentile=strategy_cfg.utility_clip_percentile,
            seed=seed,
            **common,
        )

    logger.warning("Unknown strategy '%s', falling back to Random", name)
    return RandomStrategy(num_to_select=num_to_select, seed=seed, **common)
