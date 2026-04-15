"""FedCSStrategy: deadline-aware client selection with per-round resource request.

Reference: Nishio & Yonetani, "Client Selection for Federated Learning with
Heterogeneous Resources", ICC 2019.

Per-round flow
--------------
1. **Resource request** — server randomly contacts ``c_fraction`` of all clients
   via a query message.  Clients reply with their device profile + dataset size.
   ``ClientState.profile`` is populated from these replies (server starts blind).

2. **Greedy selection** — the deadline-aware greedy algorithm (Algorithm 3 in the
   paper) picks as many responding candidates as fit within ``round_deadline_s``.

3. **Training** — selected clients receive the global model and train locally.

4. **Aggregation** — FedAvg + per-round metrics including
   ``strategy/num_within_deadline``.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from flwr.common import ArrayRecord, Message, MetricRecord
from flwr.server import Grid

from .base import BaseStrategy
from ..selection.base import ClientState

logger = logging.getLogger(__name__)


class FedCSStrategy(BaseStrategy):
    """FedCS: resource-request + deadline-aware greedy selection.

    Args:
        num_to_select: Maximum number of clients to train each round (K).
        round_deadline_s: Per-round wall-clock deadline in seconds (T).
        model_size_kb: Estimated model update size in kilobytes.
        local_epochs: Number of local training epochs per round.
        c_fraction: Fraction of all clients contacted for resource request each
            round.  The paper uses 0.5 as default.
        exploration_fraction: Fraction of K reserved for random exploration of
            clients not yet profiled.  Default 0.1.
        seed: Random seed (used for candidate sampling and exploration).
        **kwargs: Forwarded to BaseStrategy (client_states, save_path, use_wandb).
    """

    def __init__(
        self,
        num_to_select: int,
        round_deadline_s: float,
        model_size_kb: float = 1000.0,
        local_epochs: int = 1,
        c_fraction: float = 0.5,
        exploration_fraction: float = 0.1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_to_select = num_to_select
        self.round_deadline_s = round_deadline_s
        self.model_size_kb = model_size_kb
        self.local_epochs = local_epochs
        self.c_fraction = c_fraction
        self.exploration_fraction = exploration_fraction
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # configure_train: resource-request + greedy selection
    # ------------------------------------------------------------------

    def configure_train(
        self,
        round_num: int,
        arrays: ArrayRecord,
        grid: Grid,
        timeout: float,
    ) -> tuple[list[int], list[Message]]:
        all_pids = list(self.client_states.keys())

        # ---- Phase 1: Resource Request --------------------------------
        num_candidates = max(
            self.num_to_select,
            int(len(all_pids) * self.c_fraction),
        )
        candidate_pids = self._rng.sample(all_pids, min(num_candidates, len(all_pids)))

        query_msgs = self._make_query_messages(candidate_pids, "resource_request")
        resource_replies = grid.send_and_receive(query_msgs, timeout=timeout)

        responded_pids: list[int] = []
        for reply in resource_replies:
            if reply.has_error():
                continue
            nid = reply.metadata.src_node_id
            pid = self._nid_to_pid.get(nid)
            if pid is None:
                continue
            cfg = reply.content[self.configrecord_key]
            self.client_states[pid].update_from_resource_reply({
                "computation_latency_ms": float(cfg.get("computation_latency_ms", 50.0)),
                "download_kbps":          float(cfg.get("download_kbps", 5000.0)),
                "upload_kbps":            float(cfg.get("upload_kbps", 3000.0)),
                "latency_ms":             float(cfg.get("latency_ms", 50.0)),
                "num_samples":            int(cfg.get("num_samples", 0)),
            })
            responded_pids.append(pid)

        logger.info(
            "Round %d: %d/%d candidates responded to resource request",
            round_num, len(responded_pids), len(candidate_pids),
        )

        if not responded_pids:
            logger.warning("Round %d: no candidates responded, skipping", round_num)
            return [], []

        # ---- Phase 2: Greedy Selection --------------------------------
        candidate_states = {p: self.client_states[p] for p in responded_pids}
        selected = self._greedy_select(candidate_states)

        logger.info(
            "Round %d: %s selected %d clients: %s",
            round_num, self._strategy_name(), len(selected), selected,
        )

        if not selected:
            logger.warning("Round %d: selector returned empty set, skipping", round_num)
            return [], []

        messages = self._make_train_messages(selected, arrays, round_num)
        return selected, messages

    # ------------------------------------------------------------------
    # aggregate_train: add FedCS-specific extra metrics
    # ------------------------------------------------------------------

    def aggregate_train(
        self,
        round_num: int,
        replies: list[Message],
        selected_pids: list[int],
        extra_metrics: dict[str, float] | None = None,
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # Peek at valid replies without calling _check_and_log_replies (to avoid
        # double-logging when super().aggregate_train calls it again).
        valid_replies = [msg for msg in replies if not msg.has_error()]
        feedbacks_preview = self.extract_feedback(valid_replies)
        num_within = sum(
            1 for fb in feedbacks_preview.values()
            if fb["duration"] <= self.round_deadline_s
        )
        fedcs_extras = {
            "strategy/deadline": self.round_deadline_s,
            "strategy/num_within_deadline": float(num_within),
        }
        if extra_metrics:
            fedcs_extras.update(extra_metrics)

        return super().aggregate_train(round_num, replies, selected_pids, fedcs_extras)

    # ------------------------------------------------------------------
    # Greedy knapsack (Algorithm 3 in FedCS paper)
    # ------------------------------------------------------------------

    def _greedy_select(self, candidate_states: dict[int, ClientState]) -> list[int]:
        """Deadline-aware greedy selection from a set of profiled candidates."""
        candidates = list(candidate_states.keys())
        self._rng.shuffle(candidates)

        explored = [c for c in candidates if candidate_states[c].explored]
        unexplored = [c for c in candidates if not candidate_states[c].explored]

        num_explore = max(1, int(self.num_to_select * self.exploration_fraction))
        num_exploit = self.num_to_select - num_explore

        # Exploitation: sort by estimated time, greedily pick within deadline
        explored_with_time = [
            (cid, self._estimate_client_time(candidate_states[cid]))
            for cid in explored
        ]
        explored_with_time.sort(key=lambda x: x[1])

        selected: list[int] = []
        for cid, est_time in explored_with_time:
            if len(selected) >= num_exploit:
                break
            if est_time <= self.round_deadline_s:
                selected.append(cid)

        # Exploration: random sample from unexplored
        num_explore_actual = min(num_explore, len(unexplored))
        if num_explore_actual > 0:
            selected.extend(self._rng.sample(unexplored, num_explore_actual))

        # Fill remaining slots from explored that fit the deadline
        remaining = self.num_to_select - len(selected)
        if remaining > 0:
            already = set(selected)
            for cid, est_time in explored_with_time:
                if cid not in already and est_time <= self.round_deadline_s:
                    selected.append(cid)
                    if len(selected) >= self.num_to_select:
                        break

        return selected[: self.num_to_select]

    def _estimate_client_time(self, state: ClientState) -> float:
        """Estimate wall-clock time for one round on a given client.

        Priority: observed duration > profile-based estimate > deadline fallback.
        """
        if state.explored and state.last_duration_s > 0:
            return state.last_duration_s
        if state.profile is not None:
            return state.profile.estimate_round_duration(
                num_samples=max(state.num_samples, 1),
                local_epochs=self.local_epochs,
                model_size_kb=self.model_size_kb,
            )
        return self.round_deadline_s  # pessimistic fallback

    def _strategy_name(self) -> str:
        return f"FedCS(T={self.round_deadline_s:.0f}s)"
