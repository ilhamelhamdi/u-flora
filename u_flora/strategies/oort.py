"""OortStrategy: utility-guided participant selection with pacer.

Reference: Lai et al., "Oort: Efficient Federated Learning via Guided
Participant Selection".

Signals used from client feedback:
- ``train_loss`` as statistical utility input
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
        pacer_window: int = 10,
        pacer_delta: float | None = None,
        initial_t: float | None = None,
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
        self.pacer_delta = pacer_delta
        self.cutoff_c = min(max(float(cutoff_c), 0.0), 1.0)
        self.max_participation_rounds = max(0, int(max_participation_rounds))
        self.utility_clip_percentile = min(
            max(float(utility_clip_percentile), 0.0), 100.0
        )

        self._rng = random.Random(seed)

        self._preferred_t: float | None = initial_t
        self._observed_durations: list[float] = []
        self._round_utility_history: list[float] = []
        self._last_utility_clip_cap: float = 0.0
        self._last_participation_excluded: int = 0

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
        
        self._update_epsilon(num_rounds=round_num)  # Decay epsilon per round

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
        exploit_k = min(k, max(0, int(round(self.epsilon * k))))
        explore_k = k - exploit_k

        selected: list[int] = []

        # Exploit non-deterministically
        if explored and exploit_k > 0:
            utilities = {
                pid: self._client_final_utility(pid, round_num) for pid in explored
            }
            utilities, clip_cap = self._clip_utilities(utilities)
            self._last_utility_clip_cap = clip_cap

            # Sort utils
            ranked_utils = sorted(utilities.items(), key=lambda x: x[1], reverse=True)

            # Cut-off
            kth_idx = min(len(ranked_utils), max(1, exploit_k)) - 1
            kth_utility = ranked_utils[kth_idx][1]
            threshold = self.cutoff_c * kth_utility
            pool = [(pid, u) for pid, u in ranked_utils if u >= threshold]

            # Sampling
            selected.extend(self._weighted_sample_without_replacement(pool, exploit_k))

        # Explore unexplored clients
        if explore_k > 0:
            selected.extend(self._sample_unexplored(unexplored, explore_k))

        if len(selected) < k:
            remaining = [pid for pid in eligible_pids if pid not in set(selected)]
            if remaining:
                additon = self._rng.sample(
                    remaining, min(k - len(selected), len(remaining))
                )
                selected.extend(additon)

        messages = self._make_train_messages(selected, arrays, round_num)
        return selected, messages

    def configure_post_training_round(self, round_num, replies, selected_pids):
        super().configure_post_training_round(round_num, replies, selected_pids)

        valid_replies = [msg for msg in replies if not msg.has_error()]
        feedbacks_preview = self.extract_feedback(valid_replies)

        round_utility = 0.0
        for pid, fb in feedbacks_preview.items():
            duration = float(fb.get("duration", 0.0))
            if duration > 0:
                self._observed_durations.append(duration)
            round_utility += self._observed_client_utility(pid, fb, round_num)

        self._round_utility_history.append(round_utility)

        if self._preferred_t is None and len(self._observed_durations) >= 3:
            self._preferred_t = float(median(self._observed_durations))
            if self.pacer_delta is None:
                self.pacer_delta = 0.05 * self._preferred_t

        self._update_pacer_if_needed()

    def log_metrics(self, round_num, replies, selected_pids, extra_metrics=None):
        round_utility = (
            self._round_utility_history[-1]
            if self._round_utility_history
            else float("nan")
        )
        oort_extras = {
            "strategy/oort_round_utility": float(round_utility),
            "strategy/oort_explored_clients": float(
                len(
                    [
                        pid
                        for pid in self.client_states
                        if self.client_states[pid].explored
                    ]
                )
            ),
            "strategy/oort_unexplored_clients": float(
                len(
                    [
                        pid
                        for pid in self.client_states
                        if not self.client_states[pid].explored
                    ]
                )
            ),
            "strategy/oort_preferred_t": float(self._preferred_t or 0.0),
            "strategy/oort_epsilon": float(self.epsilon),
            "strategy/oort_participation_cap": float(self.max_participation_rounds),
            "strategy/oort_excluded_by_participation": float(
                self._last_participation_excluded
            ),
            "strategy/oort_utility_clip_percentile": float(
                self.utility_clip_percentile
            ),
            "strategy/oort_utility_clip_cap": float(self._last_utility_clip_cap),
        }
        if self.pacer_delta is not None:
            oort_extras["strategy/oort_pacer_delta"] = float(self.pacer_delta)

        if extra_metrics:
            oort_extras.update(extra_metrics)
        return super().log_metrics(round_num, replies, selected_pids, oort_extras)

    def _observed_client_utility(
        self, pid: int, fb: dict[str, float], round_num: int
    ) -> float:
        train_loss = float(fb.get("train_loss", 0.0))
        num_examples = float(fb.get("num_samples", 0.0))
        duration = float(fb.get("duration", 0.0))

        stat_utility = num_examples * abs(train_loss)
        staleness_bonus = self._staleness_bonus(pid, round_num)
        penalty = self._system_penalty(duration)
        return (stat_utility + staleness_bonus) * penalty

    def _client_final_utility(self, pid: int, round_num: int) -> float:
        state = self.client_states[pid]

        loss = (
            state.last_train_loss if state.last_train_loss is not None else 0.0
        )  # if no feedback yet or in initial round, assume 0 loss (could also use a default or skip)
        stat_utility = state.num_samples * abs(loss)
        staleness_bonus = self._staleness_bonus(pid, round_num)
        system_penalty = self._system_penalty(state.last_duration_s)
        return (stat_utility + staleness_bonus) * system_penalty

    def _staleness_bonus(self, pid: int, round_num: int) -> float:
        if round_num <= 1:
            return 0.0

        state = self.client_states[pid]
        last_round = max(1, state.last_selected_round)

        return math.sqrt(max(0.0, 0.1 * math.log(max(2, round_num)) / last_round))

    def _system_penalty(self, duration: float) -> float:
        if duration <= 0:
            return 1.0
        if self._preferred_t is None or duration <= self._preferred_t:
            return 1.0
        return (self._preferred_t / duration) ** self.alpha

    def _update_pacer_if_needed(self) -> None:
        if len(self._round_utility_history) < 2 * self.pacer_window:
            return
        if self._preferred_t is None:
            return

        recent = self._round_utility_history[-self.pacer_window :]
        previous = self._round_utility_history[
            -2 * self.pacer_window : -self.pacer_window
        ]
        if sum(recent) < sum(previous):
            delta = (
                self.pacer_delta
                if self.pacer_delta is not None
                else 0.05 * self._preferred_t
            )
            self._preferred_t += float(delta)

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
                w = 1 / max(1e-6, est)
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

    def _clip_utilities(
        self, utilities: dict[int, float]
    ) -> tuple[dict[int, float], float]:
        if not utilities:
            return {}, 0.0

        vals = np.array(list(utilities.values()), dtype=float)
        cap = float(np.percentile(vals, self.utility_clip_percentile))
        clipped = {pid: min(float(u), cap) for pid, u in utilities.items()}
        return clipped, cap

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

    def _strategy_name(self) -> str:
        return (
            f"Oort(K={self.num_to_select},eps={self.epsilon:.2f},"
            f"alpha={self.alpha:.1f})"
        )
