import logging

from omegaconf import DictConfig

from . import BaseStrategy, RandomStrategy, FedCSStrategy, TiFLStrategy, OortStrategy
from ..client_profile.typing import ClientState

logger = logging.getLogger(__name__)


def build_strategy(
    cfg: DictConfig,
    client_states: dict[int, ClientState],
    save_path: str,
    model_size_kb: float,
    use_wandb: bool = True,
    metric_name: str = "accuracy",
    is_higher_better: bool = True,
) -> BaseStrategy:
    """Instantiate the strategy-specific class.

    Reads from ``cfg.strategy`` (name: ``random`` | ``fedcs`` | ``tifl`` | ``oort``).
    """
    strategy = cfg.strategy
    strategy_name = strategy.name
    num_to_select = strategy.num_to_select
    seed = strategy.get("seed", 42)
    local_epochs = cfg.train.training_arguments.num_train_epochs
    heartbeat_timeout_s = cfg.server.heartbeat_timeout_s

    common = dict(
        client_states=client_states,
        save_path=save_path,
        use_wandb=use_wandb,
        metric_name=metric_name,
        is_higher_better=is_higher_better,
        model_size_kb=model_size_kb,
        heartbeat_timeout_s=heartbeat_timeout_s,
    )

    if strategy_name == "random":
        return RandomStrategy(num_to_select=num_to_select, seed=seed, **common)

    if strategy_name == "fedcs":
        return FedCSStrategy(
            round_deadline_s=strategy.fedcs.round_deadline_s,
            local_epochs=local_epochs,
            c_fraction=strategy.fedcs.c_fraction,
            seed=seed,
            **common,
        )

    if strategy_name == "tifl":
        return TiFLStrategy(
            num_to_select=num_to_select,
            num_tiers=strategy.tifl.num_tiers,
            sync_rounds=strategy.tifl.sync_rounds,
            prob_update_interval=strategy.tifl.prob_update_interval,
            round_deadline_s=strategy.tifl.round_deadline_s,
            local_epochs=local_epochs,
            seed=seed,
            **common,
        )

    if strategy_name == "oort":
        return OortStrategy(
            num_to_select=num_to_select,
            epsilon=strategy.oort.epsilon,
            alpha=strategy.oort.alpha,
            pacer_window=strategy.oort.pacer_window,
            pacer_delta=strategy.oort.pacer_delta,
            initial_t=strategy.oort.initial_t,
            cutoff_c=strategy.oort.cutoff_c,
            max_participation_rounds=strategy.oort.max_participation_rounds,
            utility_clip_percentile=strategy.oort.utility_clip_percentile,
            seed=seed,
            **common,
        )

    logger.warning("Unknown strategy '%s', falling back to Random", strategy_name)
    return RandomStrategy(num_to_select=num_to_select, seed=seed, **common)
