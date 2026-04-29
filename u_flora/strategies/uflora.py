"""UFloraStrategy: mandatory-optional client selection with async profiling and λ(t).

Design
------
Built on top of Oort's framework, U-FLoRA introduces a two-tier client selection mechanism to improve convergence rate while maintaining fairness.
Each round is split into two tiers of clients:

- Mandatory set (U): Consist of min_k clients from the explored pool. Act as the core participants that server committed to wait. The slowest client in this set determines the round duration (T_hat). Scored by Oort utility x (1 + λ(t) x participation_debt).

- Optional set: Consist of up to d_max clients, filtered by T_hat (slowest mandatory client). Aggregated only if their realised duration ≤ T_realised (max realised duration of U members). This guarantees no optional client can extend the round under jitter.

Two-threshold model
-------------------
| Threshold   | Computation                                     | Used for                        |
|-------------|-------------------------------------------------|---------------------------------|
| T_hat       | max(estimated_duration(pid) for pid in U)       | Pre-dispatch filter (optional)  |
| T_realised  | max(realised_duration(pid) for pid in U)        | Aggregation filter              |
| hard_cap    | self.round_timeout_s                            | Runaway protection (all clients)|

Async profiling (p_max > 0)
----------------------------
Before scoring each round, up to `p_max` unprofiled clients receive a `resource_request` query (near zero simulated cost). Their profile populates `ClientState.profile`, making them eligible as profiled_only candidates for the optional set. This allows faster discovery of good clients without waiting for them to be selected and trained.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

from flwr.common import ArrayRecord, Message, MetricRecord
from flwr.server import Grid

from .base import BaseStrategy
from ._fairness import ConvergenceFairnessTracker
from ..utils.timing import estimate_round_duration_s

logger = logging.getLogger(__name__)


class UFloraStrategy(BaseStrategy):
    """U-Flora: mandatory-optional FL client selection with convergence-driven fairness."""

    def __init__(
        self,
        min_k: int = 5,
        d_max: int = 5,
        p_max: int = 0,
        oort_alpha: float = 2.0,
        utility_clip_percentile: float = 95.0,
        lambda_max: float = 1.0,
        dm_ema_beta: float = 0.3,
        alpha_ema: float = 0.3,
        local_epochs: int = 1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_k = max(1, int(min_k))
        self.d_max = max(0, int(d_max))
        self.p_max = max(0, int(p_max))
        self._oort_alpha = float(oort_alpha)
        self.utility_clip_percentile = float(utility_clip_percentile)
        self.alpha_ema = float(alpha_ema)
        self._local_epochs = int(local_epochs)

        self._fairness = ConvergenceFairnessTracker(
            lambda_max=lambda_max,
            dm_ema_beta=dm_ema_beta,
        )
        self._rng = random.Random(seed)

        # Oort pacer state, used for system penalty
        self._preferred_t: float | None = None
        self._observed_durations: list[float] = []

        # Per-round state (reset each configure_train)
        self._round_U: set[int] = set()
        self._round_T_hat: float = 0.0
        self._round_optional_dispatched: int = 0

        # Per-round state (set in aggregate_train, read by log_metrics)
        self._included_pids: set[int] = set()
        self._round_T_realised: float = 0.0
        self._round_K_actual: int = 0
        self._round_optional_included: int = 0
        self._round_late_drop_U: int = 0
        self._round_late_drop_optional: int = 0
        self._round_profiled_this_round: int = 0

    # ------------------------------------------------------------------
    # Asynchrounous profiling
    # Ideally this should be run in a separate thread/process (async to the main round loop) and maintain its own pacing to continuously profile clients in the background. For simplicity, we run it synchronously at the start of configure_train, which still allows us to profile some clients before scoring and selection (zero simulated cost).
    # ------------------------------------------------------------------
    def async_profile_clients(self, round_num: int, grid: Grid) -> None:
        avail = [p for p, s in self.client_states.items() if s.available]
        self._round_profiled_this_round = 0
        if self.p_max > 0:
            unprofiled = [
                pid for pid in avail if self.client_states[pid].profile is None
            ]
            num_to_profile = min(self.p_max, len(unprofiled))
            targets = self._rng.sample(unprofiled, num_to_profile)
            if targets:
                messages = self._make_query_messages(targets, "resource_request")
                replies = grid.send_and_receive(
                    messages, timeout=self.heartbeat_timeout_s
                )
                for r in replies:
                    if r.has_error():
                        continue
                    nid = r.metadata.src_node_id
                    pid = self._nid_to_pid.get(nid)
                    if pid is None:
                        continue
                    resource = r.content[self.configrecord_key]
                    self.client_states[pid].update_from_resource_reply(
                        {
                            "computation": float(resource.get("computation", 0.0)),
                            "communication": float(resource.get("communication", 0.0)),
                            "num_samples": int(resource.get("num_samples", 0)),
                        }
                    )
                    self._round_profiled_this_round += 1
                logger.info(
                    "Round %d: U-Flora async profiling: %d/%d targets responded",
                    round_num,
                    self._round_profiled_this_round,
                    len(targets),
                )

    # ------------------------------------------------------------------
    # configure_train
    # ------------------------------------------------------------------

    def configure_train(
        self,
        round_num: int,
        arrays: ArrayRecord,
        grid: Grid,
        timeout: float,
    ) -> tuple[list[int], list[Message]]:
        avail = [p for p, s in self.client_states.items() if s.available]
        if not avail:
            return [], []

        lambda_t = self._fairness.lambda_t()

        # ---- Phase 0: Async profiling (zero simulated cost) ----
        self.async_profile_clients(round_num, grid)

        # ---- Phase 1: Partition available clients ----
        explored = [pid for pid in avail if self.client_states[pid].explored]
        profiled_only = [
            pid
            for pid in avail
            if self.client_states[pid].profile is not None
            and not self.client_states[pid].explored
        ]
        unexplored_raw = [
            pid
            for pid in avail
            if not self.client_states[pid].explored
            and self.client_states[pid].profile is None
        ]

        # ---- Phase 2: Score explored clients (strong score) ----
        score_strong: dict[int, float] = {}
        for pid in explored:
            u = self._oort_utility(pid, round_num)
            debt = self.client_states[pid].participation_debt
            score_strong[pid] = u * (1.0 + lambda_t * debt)

        # ---- Phase 3: Build mandatory set (U) ----
        ranked_strong = sorted(score_strong.items(), key=lambda x: x[1], reverse=True)
        U: list[int] = [pid for pid, _ in ranked_strong[: self.min_k]]

        # Backfill with profiled_only if not enough explored
        if len(U) < self.min_k and profiled_only:
            score_weak_po = {
                pid: self._weak_score(pid, lambda_t) for pid in profiled_only
            }
            ranked_weak = sorted(
                score_weak_po.items(), key=lambda x: x[1], reverse=True
            )
            for pid, _ in ranked_weak:
                if len(U) >= self.min_k:
                    break
                U.append(pid)

        # Backfill with unexplored (random) if still short
        if len(U) < self.min_k and unexplored_raw:
            n_need = self.min_k - len(U)
            extras = self._rng.sample(unexplored_raw, min(n_need, len(unexplored_raw)))
            U.extend(extras)

        self._round_U = set(U)

        # ---- Phase 4: T_hat from mandatory set ----
        if U:
            T_hat = max(self._estimated_full_round_s(pid) for pid in U)
        else:
            T_hat = self.round_timeout_s
        self._round_T_hat = T_hat

        # ---- Phase 5: Optional candidates (d_max, filtered by T_hat) ----
        U_set = set(U)
        remaining_strong = [
            (pid, score) for pid, score in ranked_strong if pid not in U_set
        ]
        score_weak_all = {
            pid: self._weak_score(pid, lambda_t)
            for pid in profiled_only
            if pid not in U_set
        }
        ranked_weak_all = sorted(
            score_weak_all.items(), key=lambda x: x[1], reverse=True
        )
        # TODO: consider mixing strong and weak candidates for optional set instead of strict explored-first then profiled-only.
        # Merge: explored tail first, profiled_only second
        candidates = remaining_strong + ranked_weak_all
        optional: list[int] = []
        for pid, _ in candidates:
            if len(optional) >= self.d_max:
                break
            if self._estimated_full_round_s(pid) <= T_hat:
                optional.append(pid)

        self._round_optional_dispatched = len(optional)

        dispatch = U + optional
        logger.info(
            "Round %d: U-Flora dispatching %d clients (U=%d, optional=%d), T_hat=%.1fs lambda_t=%.3f",
            round_num,
            len(dispatch),
            len(U),
            len(optional),
            T_hat,
            lambda_t,
        )

        messages = self._make_train_messages(dispatch, arrays, round_num)
        return dispatch, messages

    # ------------------------------------------------------------------
    # aggregate_train — filter optional by T_realised
    # ------------------------------------------------------------------

    def aggregate_train(
        self,
        round_num: int,
        replies: list[Message],
    ):
        hard_cap = self.round_timeout_s
        U = self._round_U

        # Extract realised durations from all replies
        feedbacks = self.extract_feedback(replies)

        # T_realised = max realised duration of U members within hard cap
        U_durations = [fb["duration"] for pid, fb in feedbacks.items() if pid in U]
        T_realised = min(max(U_durations), hard_cap)

        # Determine which clients to include in aggregation
        include: set[int] = set()
        n_late_drop_U = 0
        n_late_drop_optional = 0
        for pid, fb in feedbacks.items():
            duration_i = fb["duration"]

            if duration_i <= T_realised:
                include.add(pid)
            elif pid in U:
                n_late_drop_U += 1
                logger.warning(
                    "Round %d: U member pid=%d dropped (realised=%.1fs > hard_cap=%.1fs)",
                    round_num,
                    pid,
                    duration_i,
                    hard_cap,
                )
            else:
                n_late_drop_optional += 1
                logger.debug(
                    "Round %d: optional pid=%d dropped (realised=%.1fs > T_realised=%.1fs)",
                    round_num,
                    pid,
                    duration_i,
                    T_realised,
                )

        # Store for configure_post_training_round and log_metrics
        self._included_pids = include
        self._round_T_realised = T_realised
        self._round_late_drop_U = n_late_drop_U
        self._round_late_drop_optional = n_late_drop_optional
        self._round_K_actual = len(include)
        self._round_optional_included = len(include - U)

        # Update pacer from U durations (for future _system_penalty estimates)
        for pid, fb in feedbacks.items():
            if pid in U and fb["duration"] <= hard_cap:
                self._observed_durations.append(fb["duration"])

        # TODO: consider make the preferred_t configured externally by developer.
        if self._preferred_t is None and len(self._observed_durations) >= 3:
            from statistics import median

            self._preferred_t = float(median(self._observed_durations))

        # Filter replies: included valid + all errors (errors handled by super)
        included_nids = {self._pid_to_nid[pid] for pid in include}
        error_replies = [r for r in replies if r.has_error()]
        valid_included = [
            r
            for r in replies
            if not r.has_error() and r.metadata.src_node_id in included_nids
        ]
        filtered = valid_included + error_replies

        return super().aggregate_train(round_num, filtered)

    # ------------------------------------------------------------------
    # configure_post_training_round — update debt after aggregation
    # ------------------------------------------------------------------

    def configure_post_training_round(
        self, round_num: int, replies: list[Message], selected_pids: list[int]
    ) -> None:
        super().configure_post_training_round(round_num, replies, selected_pids)
        included = list(self._included_pids)
        self._update_participation_debt(included)

    # ------------------------------------------------------------------
    # _update_best_metric — hook λ(t) tracker
    # ------------------------------------------------------------------

    def _update_best_metric(self, round_num: int, metrics: MetricRecord) -> None:
        super()._update_best_metric(round_num, metrics)
        metric = metrics.get(f"eval_{self.metric_name}")
        if metric is not None:
            self._fairness.update(float(metric))

    # ------------------------------------------------------------------
    # log_metrics
    # ------------------------------------------------------------------

    def log_metrics(self, round_num, replies, selected_pids, extra_metrics=None):
        n_profiled = sum(
            1 for s in self.client_states.values() if s.profile is not None
        )
        n_explored = sum(1 for s in self.client_states.values() if s.explored)
        uflora_extras = {
            "strategy/uflora_lambda_t": self._fairness.lambda_t(),
            "strategy/uflora_T_hat": float(self._round_T_hat),
            "strategy/uflora_T_realised": float(self._round_T_realised),
            "strategy/uflora_K_actual": float(self._round_K_actual),
            "strategy/uflora_optional_dispatched": float(
                self._round_optional_dispatched
            ),
            "strategy/uflora_optional_included": float(self._round_optional_included),
            "strategy/uflora_late_drop_optional": float(self._round_late_drop_optional),
            "strategy/uflora_late_drop_U": float(self._round_late_drop_U),
            "strategy/uflora_profiled_clients": float(n_profiled),
            "strategy/uflora_explored_clients": float(n_explored),
        }
        if extra_metrics:
            uflora_extras.update(extra_metrics)
        return super().log_metrics(round_num, replies, selected_pids, uflora_extras)

    # ------------------------------------------------------------------
    # Utility helpers (Oort-style, standalone)
    # ------------------------------------------------------------------

    def _oort_utility(self, pid: int, round_num: int) -> float:
        state = self.client_states[pid]
        loss = state.last_train_loss if state.last_train_loss is not None else 0.0
        stat_utility = state.num_samples * abs(loss)
        staleness = self._staleness_bonus(pid, round_num)
        penalty = self._system_penalty(state.last_duration_s)
        return (stat_utility + staleness) * penalty

    def _staleness_bonus(self, pid: int, round_num: int) -> float:
        if round_num <= 1:
            return 0.0
        last_round = max(1, self.client_states[pid].last_selected_round)
        return math.sqrt(0.1 * math.log(round_num) / last_round)

    def _system_penalty(self, duration: float) -> float:
        if self._preferred_t is None or duration <= self._preferred_t:
            return 1.0
        return (self._preferred_t / duration) ** self._oort_alpha

    def _weak_score(self, pid: int, lambda_t: float) -> float:
        """Score for profiled-only (not yet explored) clients."""
        state = self.client_states[pid]
        d_hat = self._estimated_full_round_s(pid)
        data_score = math.log(max(state.num_samples, 1))
        system_score = (1.0 / d_hat) ** self._oort_alpha
        debt = state.participation_debt
        return data_score * system_score * (1.0 + lambda_t * debt)

    def _estimated_full_round_s(self, pid: int) -> float:
        """Return estimated round duration for given client pid based on their profile."""
        state = self.client_states[pid]
        if state.last_duration_s is not None:
            return float(state.last_duration_s)
        if state.profile is not None:
            _, _, total = estimate_round_duration_s(
                profile={
                    "computation": state.profile.computation,
                    "communication": state.profile.communication,
                },
                num_samples=max(state.num_samples, 1),
                local_epochs=self._local_epochs,
                model_size_kb=self.model_size_kb,
            )
            return total
        return float("inf")

    # ------------------------------------------------------------------
    # Participation debt
    # ------------------------------------------------------------------

    def _update_participation_debt(self, included_pids: list[int]) -> None:
        n_avail = max(1, len(self._last_available_pids))
        fair_share = len(included_pids) / n_avail
        included_set = set(included_pids)
        for pid in self._last_available_pids:
            delta = fair_share - (1.0 if pid in included_set else 0.0)
            self.client_states[pid].participation_debt += delta

    def _strategy_name(self) -> str:
        return (
            f"UFlora(K={self.min_k}+{self.d_max},p={self.p_max},"
            f"lam={self._fairness.lambda_max:.1f})"
        )
