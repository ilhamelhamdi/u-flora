"""OortStrategy: utility-guided participant selection with pacer.

Reference: Lai et al., "Oort: Efficient Federated Learning via Guided
Participant Selection".

Signals used from client feedback:
- ``train_loss_rms`` as statistical utility input
- ``duration`` as system-speed signal (D(i))
- ``num-examples`` as local batch cardinality proxy (|B_i|)
"""

from __future__ import annotations

import logging
import math
import random
from statistics import median
from typing import Any

import numpy as np

from flwr.common import ArrayRecord, Message, MetricRecord
from flwr.server import Grid

from .base import BaseStrategy

logger = logging.getLogger(__name__)


class OortStrategy(BaseStrategy):
    """Oort participant selection with utility, staleness, and pacer."""

    def __init__(
        self,
        num_to_select: int,
        epsilon: float = 0.9,  # same as in paper
        epsilon_decay: float = 0.98,  # per-round decay factor for epsilon
        epsilon_min: float = 0.2,  # minimum epsilon for exploration
        alpha: float = 2.0,
        pacer_window: int = 10,  # W
        pacer_delta: float | None = None,  # ∆
        t_mode: str = "percentile",  # "percentile" (Oort codebase) | "fixed" (Oort paper)
        round_threshold_init: float = 20.0,  # percentile in [0,100]; percentile mode
        initial_t: float | None = None,  # absolute seconds; fixed mode only
        cutoff_c: float = 0.95,
        max_participation_rounds: int = 20,
        utility_clip_percentile: float = 95.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_to_select = num_to_select

        self.epsilon = min(max(float(epsilon), 0.0), 1.0)
        self.epsilon_decay = min(max(float(epsilon_decay), 0.0), 1.0)
        self.epsilon_min = min(max(float(epsilon_min), 0.0), 1.0)
        self.alpha = float(alpha)
        self.pacer_window = max(1, int(pacer_window))
        self.cutoff_c = min(max(float(cutoff_c), 0.0), 1.0)
        self.max_participation_rounds = max(0, int(max_participation_rounds))
        self.utility_clip_percentile = min(
            max(float(utility_clip_percentile), 0.0), 100.0
        )

        self._rng = random.Random(seed)

        if t_mode not in ("percentile", "fixed"):
            raise ValueError(f"t_mode must be 'percentile' or 'fixed', got {t_mode!r}")
        self.t_mode = t_mode
        self._round_threshold = min(max(float(round_threshold_init), 0.0), 100.0)
        self._preferred_t: float | None = (
            float(initial_t) if (t_mode == "fixed" and initial_t is not None) else None
        )
        if t_mode == "fixed" and self._preferred_t is None:
            raise ValueError("t_mode='fixed' requires initial_t to be set")
        self.pacer_delta = (
            float(pacer_delta)
            if pacer_delta is not None
            else (5.0 if t_mode == "percentile" else 0.0)
        )

        self._observed_durations: list[float] = []
        self._round_utility_history: list[float] = []
        self._last_utility_clip_cap: float = 0.0
        self._last_participation_excluded: int = 0
        self._last_exploit_k: int = 0
        self._last_explore_k: int = 0

    # ======================================================================
    # Selection
    # ======================================================================

    def configure_train(
        self,
        round_num: int,
        arrays: ArrayRecord,
        grid: Grid,
        timeout: float,
    ) -> tuple[list[int], list[Message]]:
        # Only consider clients reported available by heartbeat this round.
        all_pids = [p for p, s in self.client_states.items() if s.available]
        if not all_pids:
            return [], []

        # Refresh the percentile-derived deadline from the current duration distribution (no-op in fixed mode).
        self._refresh_preferred_t()

        # Filter out clients which are selected more than `max_participation_rounds`
        eligible_pids = self._filter_by_participation_cap(all_pids)
        if not eligible_pids:
            logger.warning(
                "Oort: no eligible clients after participation cap; falling back to all clients"
            )
            eligible_pids = all_pids

        explored = [pid for pid in eligible_pids if self.client_states[pid].explored]
        unexplored = [
            pid for pid in eligible_pids if not self.client_states[pid].explored
        ]

        k = min(self.num_to_select, len(eligible_pids))
        explore_k = min(k, max(0, int(round(self.epsilon * k))))
        exploit_k = k - explore_k

        self._last_exploit_k = exploit_k
        self._last_explore_k = explore_k

        selected: list[int] = []

        # ---- Exploitation: normalized utility + staleness + system penalty ----
        if explored and exploit_k > 0:
            utilities = self._compute_exploit_utilities(explored, round_num)

            ranked_utils = sorted(utilities.items(), key=lambda x: x[1], reverse=True)

            kth_idx = min(len(ranked_utils), max(1, exploit_k)) - 1
            kth_utility = ranked_utils[kth_idx][1]
            threshold = self.cutoff_c * kth_utility
            pool = [(pid, u) for pid, u in ranked_utils if u >= threshold]

            selected.extend(self._weighted_sample_without_replacement(pool, exploit_k))

        # ---- Exploration: sample unexplored by speed --------------------------
        if explore_k > 0:
            selected.extend(self._sample_unexplored(unexplored, explore_k))

        # ---- Top-up if short --------------------------------------------------
        if len(selected) < k:
            remaining = [pid for pid in eligible_pids if pid not in set(selected)]
            if remaining:
                additon = self._rng.sample(
                    remaining, min(k - len(selected), len(remaining))
                )
                selected.extend(additon)

        messages = self._make_train_messages(selected, arrays, round_num)
        return selected, messages

    def _compute_exploit_utilities(
        self, explored: list[int], round_num: int
    ) -> dict[int, float]:
        """Paper line 10-12 / codebase getTopK utility pipeline.

        raw stat-utility -> clip at 95th pct -> min-max normalize ->
        + staleness (temporal uncertainty) -> * system penalty.
        """
        raw: dict[int, float] = {}
        for pid in explored:
            s = self.client_states[pid]
            loss_rms = (
                s.last_train_loss_rms if s.last_train_loss_rms is not None else 0.0
            )
            raw[pid] = float(s.num_samples) * abs(float(loss_rms))

        if not raw:
            return {}

        # Clip raw statistical utility at the 95th percentile.
        vals = np.array(list(raw.values()), dtype=float)
        clip_cap = float(np.percentile(vals, self.utility_clip_percentile))
        self._last_utility_clip_cap = clip_cap
        clipped = {pid: min(v, clip_cap) for pid, v in raw.items()}

        # Min-max normalize to ~[0,1] so the staleness term is on the same scale.
        cvals = list(clipped.values())
        cmin = min(cvals)
        crange = max(max(cvals) - cmin, 1e-4)

        util: dict[int, float] = {}
        for pid, v in clipped.items():
            score = (v - cmin) / crange
            score += self._staleness_bonus(pid, round_num)  # temporal uncertainty
            score *= self._system_penalty(self.client_states[pid].last_duration_s)
            util[pid] = score
        return util

    # ======================================================================
    # Post-round bookkeeping + pacer
    # ======================================================================

    def configure_post_training_round(self, round_num, replies, selected_pids):
        super().configure_post_training_round(round_num, replies, selected_pids)

        valid_replies = [msg for msg in replies if not msg.has_error()]
        feedbacks_preview = self.extract_feedback(valid_replies)

        round_utility = 0.0
        for pid, fb in feedbacks_preview.items():
            round_utility += self._observed_client_stat_utility(pid, fb)
        self._round_utility_history.append(round_utility)

        self._update_epsilon(num_rounds=round_num)  # Decay epsilon per round
        self._update_pacer_if_needed(round_num)

    def _refresh_preferred_t(self) -> None:
        """Recompute T as the round_threshold percentile of live durations."""
        if self.t_mode == "fixed":
            return  # constant T = initial_t
        durations = sorted(
            float(s.last_duration_s)
            for s in self.client_states.values()
            if s.last_duration_s is not None and s.last_duration_s > 0
        )
        if not durations:
            self._preferred_t = None  # no penalty until we observe durations
            return
        if self._round_threshold >= 100.0:
            self._preferred_t = float("inf")  # penalty disabled
            return
        idx = min(
            int(len(durations) * self._round_threshold / 100.0), len(durations) - 1
        )
        self._preferred_t = float(durations[idx])

    def _update_pacer_if_needed(self, round_num: int) -> None:
        """Relax the deadline when statistical utility stops improving."""
        if self.t_mode == "fixed":
            return
        if len(self._round_utility_history) < 2 * self.pacer_window:
            return
        if round_num % self.pacer_window != 0:
            return

        recent = sum(self._round_utility_history[-self.pacer_window :])
        previous = sum(
            self._round_utility_history[-2 * self.pacer_window : -self.pacer_window]
        )

        # One-sided relaxation (paper direction).
        if recent <= previous:
            self._round_threshold = min(100.0, self._round_threshold + self.pacer_delta)
        # --- Optional codebase two-sided variant (re-tighten on sharp gains) ---
        # elif recent >= previous * 6.0:  # |Δ| >= 5x previous
        #     self._round_threshold = max(
        #         self.pacer_delta, self._round_threshold - self.pacer_delta
        #     )

    # ======================================================================
    # Utility components
    # ======================================================================

    def _observed_client_stat_utility(self, pid: int, fb: dict[str, float]) -> float:
        train_loss_rms = float(fb.get("train_loss_rms", 0.0))
        num_examples = float(fb.get("num_samples", 0.0))
        stat_utility = num_examples * abs(train_loss_rms)
        return stat_utility

    def _staleness_bonus(self, pid: int, round_num: int) -> float:
        if round_num <= 1:
            return 0.0
        state = self.client_states[pid]
        last_round = max(1, state.last_selected_round)
        return math.sqrt(max(0.0, 0.1 * math.log(max(2, round_num)) / last_round))

    def _system_penalty(self, duration: float | None) -> float:
        if duration is None or duration <= 0:
            return 1.0
        if self._preferred_t is None or duration <= self._preferred_t:
            return 1.0
        return (self._preferred_t / duration) ** self.alpha

    # ======================================================================
    # Exploration / sampling helpers
    # ======================================================================

    def _sample_unexplored(self, unexplored: list[int], count: int) -> list[int]:
        if count <= 0 or not unexplored:
            return []

        weighted_pool: list[tuple[int, float]] = []
        for pid in unexplored:
            state = self.client_states[pid]
            if state.profile is not None:
                _, _, est = self._estimate_duration(
                    state.profile,
                    num_samples=max(1, state.num_samples),
                    local_epochs=1,
                )
                w = 1 / max(1e-6, est)  # faster clients preferred (SampleBySpeed)
            else:
                w = 1
            weighted_pool.append((pid, w))

        return self._weighted_sample_without_replacement(weighted_pool, count)

    def _filter_by_participation_cap(self, client_ids: list[int]) -> list[int]:
        if self.max_participation_rounds <= 0:
            self._last_participation_excluded = 0
            return list(client_ids)
        eligible = [
            pid
            for pid in client_ids
            if self.client_states[pid].times_selected < self.max_participation_rounds
        ]
        self._last_participation_excluded = len(client_ids) - len(eligible)
        return eligible

    def _weighted_sample_without_replacement(
        self,
        weighted_items: list[tuple[int, float]],
        k: int,
    ) -> list[int]:
        if k <= 0 or not weighted_items:
            return []
        items = [(pid, max(0.0, float(w))) for pid, w in weighted_items]
        chosen: list[int] = []

        for _ in range(min(k, len(items))):
            total_w = sum(w for _, w in items)

            # Fallback to equal random sampling
            if total_w <= 0:
                remaining = [pid for pid, _ in items]
                chosen.extend(
                    self._rng.sample(remaining, min(k - len(chosen), len(remaining)))
                )
                break
            r = self._rng.random() * total_w
            acc = 0.0
            idx = 0
            for i, (_, w) in enumerate(items):
                acc += w
                if acc >= r:
                    idx = i
                    break
            pid, _ = items.pop(idx)
            chosen.append(pid)
        return chosen

    def _update_epsilon(self, num_rounds: int) -> None:
        if num_rounds <= 1:
            return
        new_epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon = new_epsilon

    # ======================================================================
    # Logging
    # ======================================================================

    def log_metrics(self, round_num, replies, selected_pids, extra_metrics=None):
        round_utility = (
            self._round_utility_history[-1]
            if self._round_utility_history
            else float("nan")
        )
        oort_extras = {
            "strategy/oort_round_utility": float(round_utility),
            "strategy/oort_preferred_t": float(self._preferred_t or 0.0),
            "strategy/oort_round_threshold": float(self._round_threshold),
            "strategy/oort_epsilon": float(self.epsilon),
            "strategy/oort_excluded_by_participation": float(
                self._last_participation_excluded
            ),
            "strategy/oort_utility_clip_cap": float(self._last_utility_clip_cap),
            "strategy/oort_exploit_k": float(self._last_exploit_k),
            "strategy/oort_explore_k": float(self._last_explore_k),
            "strategy/oort_pacer_delta": float(self.pacer_delta),
            "strategy/oort_explored_clients": float(
                len([p for p in self.client_states if self.client_states[p].explored])
            ),
            "strategy/oort_unexplored_clients": float(
                len(
                    [
                        p
                        for p in self.client_states
                        if not self.client_states[p].explored
                    ]
                )
            ),
        }

        if extra_metrics:
            oort_extras.update(extra_metrics)
        return super().log_metrics(round_num, replies, selected_pids, oort_extras)

    def _strategy_name(self) -> str:
        return (
            f"Oort(K={self.num_to_select},eps={self.epsilon:.2f},"
            f"alpha={self.alpha:.1f})"
        )
