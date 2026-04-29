"""OortFairStrategy: Oort + convergence-driven fairness λ(t).

Adds two mechanisms on top of OortStrategy:

1. Participation debt — each available client accumulates debt proportional to how much it is under-selected relative to its fair share. Debt is applied as a multiplicative bonus on top of the base Oort utility, so chronically skipped clients get a second-chance uplift.

2. lambda(t) — the debt multiplier is scaled by λ(t), which tracks smoothed convergence speed. When convergence stalls, lambda(t) rises (more fairness pressure). When convergence accelerates, λ drops (utility dominates).

Only three methods are overridden from OortStrategy; everything else is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from flwr.common import MetricRecord

from .oort import OortStrategy
from ._fairness import ConvergenceFairnessTracker

logger = logging.getLogger(__name__)


class OortFairStrategy(OortStrategy):
    """Oort participant selection augmented with convergence-driven fairness."""

    def __init__(
        self,
        lambda_max: float = 1.0,
        dm_ema_beta: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._fairness = ConvergenceFairnessTracker(
            lambda_max=lambda_max,
            dm_ema_beta=dm_ema_beta,
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _client_final_utility(self, pid: int, round_num: int) -> float:
        base = super()._client_final_utility(pid, round_num)
        debt = self.client_states[pid].participation_debt
        return base * (1.0 + self._fairness.lambda_t() * debt)

    def configure_post_training_round(
        self, round_num: int, replies, selected_pids: list[int]
    ) -> None:
        super().configure_post_training_round(round_num, replies, selected_pids)
        self._update_participation_debt(selected_pids)

    def _update_best_metric(self, round_num: int, metrics: MetricRecord) -> None:
        super()._update_best_metric(round_num, metrics)
        metric = metrics.get(f"eval_{self.metric_name}")
        if metric is not None:
            self._fairness.update(float(metric))

    # ------------------------------------------------------------------
    # Debt helper
    # ------------------------------------------------------------------

    def _update_participation_debt(self, selected_pids: list[int]) -> None:
        n_avail = max(1, len(self._last_available_pids))
        fair_share = len(selected_pids) / n_avail
        selected_set = set(selected_pids)
        for pid in self._last_available_pids:
            delta = fair_share - (1.0 if pid in selected_set else 0.0)
            self.client_states[pid].participation_debt += delta

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metrics(self, round_num, replies, selected_pids, extra_metrics=None):
        all_debts = [
            self.client_states[pid].participation_debt for pid in self.client_states
        ]
        mean_debt = sum(all_debts) / max(1, len(all_debts))
        oort_fair_extras = {
            "strategy/oort_fair_lambda_t": self._fairness.lambda_t(),
            "strategy/oort_fair_running_max_dm": self._fairness.running_max_dm,
            "strategy/oort_fair_mean_debt": mean_debt,
        }
        if extra_metrics:
            oort_fair_extras.update(extra_metrics)
        return super().log_metrics(round_num, replies, selected_pids, oort_fair_extras)

    def _strategy_name(self) -> str:
        return f"OortFair(K={self.num_to_select},lam={self._fairness.lambda_max:.1f})"
