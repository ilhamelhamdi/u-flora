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
from ..client_profile.typing import ClientState

logger = logging.getLogger(__name__)


class FedCSStrategy(BaseStrategy):
    """FedCS: resource-request + deadline-aware greedy selection.

    Args:
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
        round_deadline_s: float,
        local_epochs: int = 1,
        c_fraction: float = 0.5,  # Fraction of clients to contact for resource request each round.
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.round_deadline_s = round_deadline_s
        self.local_epochs = local_epochs
        self.c_fraction = c_fraction
        self._rng = random.Random(seed)

    def start(
        self,
        grid,
        initial_arrays,
        num_rounds,
        timeout=None,
        train_config=None,
        evaluate_config=None,
        evaluate_fn=None,
    ):
        return super().start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            timeout=self.round_deadline_s,  # Inject round deadline as timeout for resource request + selection phase
            train_config=train_config,
            evaluate_config=evaluate_config,
            evaluate_fn=evaluate_fn,
        )

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
        # Only consider available clients (per heartbeat this round).
        all_pids = [p for p, s in self.client_states.items() if s.available]
        if not all_pids:
            return [], []

        # ---- Phase 1: Resource Request in Every Round --------------------------------
        num_candidates = int(len(all_pids) * self.c_fraction)
        candidate_pids = self._rng.sample(all_pids, min(num_candidates, len(all_pids)))

        query_msgs = self._make_query_messages(candidate_pids, "resource_request")
        resource_replies = grid.send_and_receive(
            query_msgs, timeout=timeout
        )  # Will be quick since it's just a profile query

        responded_pids: list[int] = []
        for reply in resource_replies:
            if reply.has_error():
                continue
            nid = reply.metadata.src_node_id
            pid = self._nid_to_pid.get(nid)
            if pid is None:
                continue
            resource = reply.content[self.configrecord_key]
            self.client_states[pid].update_from_resource_reply(
                {
                    "computation": float(resource.get("computation", 0.0)),
                    "communication": float(resource.get("communication", 0.0)),
                    "num_samples": int(resource.get("num_samples", 0)),
                }
            )
            responded_pids.append(pid)

        logger.info(
            "Round %d: %d/%d candidates responded to resource request",
            round_num,
            len(responded_pids),
            len(candidate_pids),
        )

        # ---- Phase 2: Greedy Selection --------------------------------
        candidate_states = {
            p: self.client_states[p] for p in responded_pids
        }  # Equal to K' in the paper
        selected = self._greedy_select(candidate_states)

        logger.info(
            "Round %d: %s selected %d clients: %s",
            round_num,
            self._strategy_name(),
            len(selected),
            selected,
        )

        messages = self._make_train_messages(selected, arrays, round_num)
        return selected, messages

    # ------------------------------------------------------------------
    # log_metrics: add FedCS-specific extra metrics
    # ------------------------------------------------------------------

    def log_metrics(self, round_num, replies, selected_pids, extra_metrics=None):
        fedcs_extras = {
            "strategy/deadline": self.round_deadline_s,
        }

        if extra_metrics:
            fedcs_extras.update(extra_metrics)

        return super().log_metrics(round_num, replies, selected_pids, fedcs_extras)

    # ------------------------------------------------------------------
    # Greedy knapsack (Algorithm 3 in FedCS paper)
    # ------------------------------------------------------------------

    def _greedy_select(self, candidate_states: dict[int, ClientState]) -> list[int]:
        selected: set[int] = set()  # S
        distribution_time: float = 0.0  # T_dist
        theta: float = (
            0.0  # Θ_i : estimated elapsed time from the beginning of the Scheduled Update and Upload step until the ki-th client completes the update and upload procedures
        )

        # Parameters from the paper that we treat as neglible or zero for simplicity:
        T_cs = 0.0 # T_cs : the overhead of client selection procedure.
        T_agg = 0.0 # T_agg : the time for aggregating the model updates from the selected clients.

        t_dist = {
            cid: self._estimate_distribution_to_client_i(candidate_states[cid])
            for cid in candidate_states
        }
        t_upload = {
            cid: self._estimate_upload_from_client_i(candidate_states[cid])
            for cid in candidate_states
        }
        t_update = {
            cid: self._estimate_training_time_on_client_i(candidate_states[cid])
            for cid in candidate_states
        }

        while len(candidate_states) > 0:
            costs = {
                cid: 
                max(distribution_time, t_dist[cid]) 
                + t_upload[cid]
                + max(0, t_update[cid] - theta)
                for cid in candidate_states
            }
            selected_cid = min(costs, key=costs.get)
            candidate_states.pop(selected_cid) # Remove from candidates
            theta_prime = theta + t_upload[selected_cid] + max(0, t_update[selected_cid] - theta)
            t = T_cs + max(distribution_time, t_dist[selected_cid]) + theta_prime + T_agg

            if t <= self.round_deadline_s:
                selected.add(selected_cid)
                distribution_time = max(distribution_time, t_dist[selected_cid])
                theta = theta_prime

        return list(selected)

    def _estimate_distribution_to_client_i(self, state: ClientState) -> float:
        """Estimate time to download the global model in second(s)."""
        comm = max(1.0, state.profile.communication)
        return self.model_size_kb / comm

    def _estimate_upload_from_client_i(self, state: ClientState) -> float:
        """Estimate time to upload the model update in second(s)."""
        comm = max(1.0, state.profile.communication)
        return self.model_size_kb / comm

    def _estimate_training_time_on_client_i(self, state: ClientState) -> float:
        """Estimate local training time on client i in second(s).

        Uses the calibration constants threaded through BaseStrategy
        (``t_min_ms`` from profile metadata, ``t_actual_ms`` per task).
        """
        compute_s, _, _ = self._estimate_duration(
            state.profile,
            num_samples=max(state.num_samples, 1),
            local_epochs=self.local_epochs,
        )
        return compute_s

    def _strategy_name(self) -> str:
        return f"FedCS(T={self.round_deadline_s:.0f}s)"
