import logging

from omegaconf import DictConfig

from . import BaseStrategy, RandomStrategy, FedCSStrategy, TiFLStrategy, OortStrategy, OortFairStrategy, UFloraStrategy
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
    seed = cfg.seed
    strategy_name = strategy.name
    num_to_select = strategy.num_to_select
    local_epochs = cfg.train.training_arguments.num_train_epochs
    heartbeat_timeout_s = cfg.server.heartbeat_timeout_s
    round_timeout_s = float(cfg.server.get("round_timeout_s", 600.0))

    common = dict(
        client_states=client_states,
        save_path=save_path,
        use_wandb=use_wandb,
        metric_name=metric_name,
        is_higher_better=is_higher_better,
        model_size_kb=model_size_kb,
        heartbeat_timeout_s=heartbeat_timeout_s,
        round_timeout_s=round_timeout_s,
    )

    if strategy_name == "random":
        return RandomStrategy(num_to_select=num_to_select, seed=seed, **common)

    if strategy_name == "fedcs":
        return FedCSStrategy(
            round_deadline_s=strategy.fedcs.round_deadline_s,
            local_epochs=local_epochs,
            c_fraction=strategy.fedcs.c_fraction,
            num_to_select=num_to_select,
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
            epsilon_decay=strategy.oort.epsilon_decay,
            epsilon_min=strategy.oort.epsilon_min,
            alpha=strategy.oort.alpha,
            pacer_window=strategy.oort.pacer_window,
            pacer_delta=strategy.oort.pacer_delta,
            t_mode=strategy.oort.t_mode,
            initial_t=strategy.oort.initial_t,
            round_threshold_init=strategy.oort.round_threshold_init,
            cutoff_c=strategy.oort.cutoff_c,
            max_participation_rounds=strategy.oort.max_participation_rounds,
            utility_clip_percentile=strategy.oort.utility_clip_percentile,
            seed=seed,
            **common,
        )

    if strategy_name == "oort_fair":
        return OortFairStrategy(
            num_to_select=num_to_select,
            epsilon=strategy.oort.epsilon,
            epsilon_decay=strategy.oort.epsilon_decay,
            epsilon_min=strategy.oort.epsilon_min,
            alpha=strategy.oort.alpha,
            pacer_window=strategy.oort.pacer_window,
            pacer_delta=strategy.oort.pacer_delta,
            initial_t=strategy.oort.initial_t,
            cutoff_c=strategy.oort.cutoff_c,
            max_participation_rounds=strategy.oort.max_participation_rounds,
            utility_clip_percentile=strategy.oort.utility_clip_percentile,
            lambda_max=strategy.oort_fair.lambda_max,
            dm_ema_beta=strategy.oort_fair.dm_ema_beta,
            seed=seed,
            **common,
        )

    if strategy_name == "uflora":
        return UFloraStrategy(
            min_k=strategy.uflora.min_k,
            d_max=strategy.uflora.d_max,
            p_max=strategy.uflora.p_max,
            oort_alpha=strategy.oort.alpha,
            utility_clip_percentile=strategy.oort.utility_clip_percentile,
            lambda_max=strategy.uflora.lambda_max,
            dm_ema_beta=strategy.uflora.dm_ema_beta,
            alpha_ema=strategy.uflora.alpha_ema,
            local_epochs=local_epochs,
            seed=seed,
            **common,
        )

    logger.warning("Unknown strategy '%s', falling back to Random", strategy_name)
    return RandomStrategy(num_to_select=num_to_select, seed=seed, **common)
