"""TiFLStrategy: tier-based client selection with profiling rounds.

Reference: Chai et al., "TiFL: A Tier-based Federated Learning System".

Implementation outline
----------------------
1. Profiling phase (before FL rounds):
   - Run ``sync_rounds`` resource-request rounds over all clients.
   - For each round, clients responding within ``round_deadline_s`` are profiled.
   - Non-responders are assigned latency ``round_deadline_s``.
   - Clients with cumulative latency >= ``sync_rounds * round_deadline_s`` are considered dropouts and excluded.
2. Tier construction:
   - Active clients are sorted by average profiling latency and split into ``num_tiers`` contiguous latency tiers.
3. Adaptive tier selection:
   - Each FL round selects one tier with weighted probability.
   - Sample ``num_to_select`` clients uniformly from the chosen tier.
   - Decrement tier credits.
   - Every ``prob_update_interval`` rounds, if selected-tier accuracy did not
     improve, update tier probabilities with ChangeProbs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import random
from typing import Any

import numpy as np
from flwr.common import ArrayRecord, Message
from flwr.server import Grid

from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class ClientProfilingAccumulator:
    """Profiling statistics for one client during TiFL sync rounds."""

    cumulative_latency: float = 0.0
    latency_samples: list[float] = field(default_factory=list)

    def add_sample(self, latency: float) -> None:
        self.cumulative_latency += float(latency)
        self.latency_samples.append(float(latency))

    def mean_latency(self, default: float) -> float:
        if not self.latency_samples:
            return float(default)
        return float(np.mean(self.latency_samples))


@dataclass
class TiFLTierState:
    """Mutable state for one TiFL latency tier."""

    tier_id: int
    clients: list[int] = field(default_factory=list)
    client_latencies: dict[int, float] = field(default_factory=dict)  # pid -> latency
    credits: int = 0
    probability: float = 0.0
    accuracy: float = 0.0
    last_checked_accuracy: float = 0.0


@dataclass
class TiFLRuntimeState:
    """All TiFL runtime state grouped in one container."""

    total_rounds: int = 0
    active_clients: list[int] = field(default_factory=list)
    dropout_clients: set[int] = field(default_factory=set)
    tiers: dict[int, TiFLTierState] = field(default_factory=dict)
    last_selected_tier: int | None = None


class TiFLStrategy(BaseStrategy):
    """TiFL strategy with profiling + tier-aware adaptive selection."""

    def __init__(
        self,
        num_to_select: int,
        num_tiers: int = 5,
        sync_rounds: int = 3,
        prob_update_interval: int = 10,
        round_deadline_s: float = 300.0,
        local_epochs: int = 1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
            # evaluate_during_training=True # No need to evaluate during training for TiFL since it uses training loss as a proxy for accuracy. This also avoids extra evaluation overhead.
        )
        self.num_to_select = num_to_select
        self.num_tiers = max(1, num_tiers)
        self.sync_rounds = max(1, sync_rounds)
        self.prob_update_interval = max(1, prob_update_interval)
        self.round_deadline_s = float(round_deadline_s)
        self.local_epochs = int(local_epochs)
        self._rng = random.Random(seed)
        self._state = TiFLRuntimeState()

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600.0,
        train_config=None,
        evaluate_config=None,
        evaluate_fn=None,
    ):
        self._state.total_rounds = int(num_rounds)
        return super().start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            timeout=timeout,
            train_config=train_config,
            evaluate_config=evaluate_config,
            evaluate_fn=evaluate_fn,
        )

    def configure_pretrain(
        self,
        grid: Grid,
        arrays: ArrayRecord,
        timeout: float,
    ) -> None:
        all_pids = list(self.client_states.keys())
        if not all_pids:
            return None

        logger.info(
            "TiFL profiling: %d rounds, deadline=%.1fs, candidates=%d",
            self.sync_rounds,
            self.round_deadline_s,
            len(all_pids),
        )

        profiling = {pid: ClientProfilingAccumulator() for pid in all_pids}

        # TODO: instead of resource-requesting, the paper suggests running actual training rounds with a small model and small local epoch. This would require more changes to the codebase but would yield more accurate profiling.
        for _ in range(self.sync_rounds):
            query_msgs = self._make_query_messages(all_pids, "resource_request")
            replies = grid.send_and_receive(query_msgs, timeout=self.round_deadline_s)

            replied: set[int] = set()
            for reply in replies:
                if reply.has_error():
                    continue
                nid = reply.metadata.src_node_id
                pid = self._nid_to_pid.get(nid)
                if pid is None:
                    continue

                cfg = reply.content[self.configrecord_key]
                profile_data = {
                    "computation": float(cfg.get("computation", 0.0)),
                    "communication": float(cfg.get("communication", 0.0)),
                    "num_samples": int(cfg.get("num_samples", 1)),
                }
                self.client_states[pid].update_from_resource_reply(profile_data)
                state = self.client_states[pid]

                if state.profile is not None:
                    _, _, latency = self._estimate_duration(
                        state.profile,
                        num_samples=max(1, state.num_samples),
                        local_epochs=self.local_epochs,
                    )
                else:
                    latency = self.round_deadline_s

                latency = min(latency, self.round_deadline_s)
                profiling[pid].add_sample(latency)
                replied.add(pid)

            for pid in all_pids:
                if pid in replied:
                    continue
                profiling[pid].add_sample(self.round_deadline_s)

        dropout_threshold = self.sync_rounds * self.round_deadline_s
        self._state.dropout_clients = {
            pid
            for pid, acc in profiling.items()
            if acc.cumulative_latency >= dropout_threshold
        }
        self._state.active_clients = [
            pid for pid in all_pids if pid not in self._state.dropout_clients
        ]

        avg_latency_profiling = {
            pid: profiling[pid].mean_latency(self.round_deadline_s)
            for pid in self._state.active_clients
        }

        self._build_tiers(avg_latency_profiling)
        self._initialize_tier_controls()
        self._log_tier_structure()

        logger.info(
            "TiFL profiling complete: active=%d dropouts=%d tiers=%d",
            len(self._state.active_clients),
            len(self._state.dropout_clients),
            len(self._state.tiers),
        )

    def configure_train(
        self,
        round_num: int,
        arrays: ArrayRecord,
        grid: Grid,
        timeout: float,
    ) -> tuple[list[int], list[Message]]:
        if not self._state.tiers:
            raise RuntimeError(
                "TiFL `configure_train` called before tiers are initialized. Run `configure_pretrain` first."
            )

        # Restrict every tier's roster to *currently available* clients
        # (per heartbeat at the start of this round).
        round_clients_by_tier: dict[int, list[int]] = {}
        for tier_id, tier in self._state.tiers.items():
            avail_in_tier = [
                pid for pid in tier.clients if self.client_states[pid].available
            ]
            round_clients_by_tier[tier_id] = avail_in_tier

        available_tiers = [
            tier_id
            for tier_id, tier in self._state.tiers.items()
            if tier.credits > 0 and round_clients_by_tier[tier_id]
        ]
        if not available_tiers:
            logger.warning(
                "TiFL: no tiers with remaining credits and available clients"
            )
            self._state.last_selected_tier = None
            return [], []

        if (
            round_num % self.prob_update_interval == 0
            and round_num >= self.prob_update_interval
        ):
            self._maybe_update_probs()

        chosen_tier = self._weighted_choose_tier(available_tiers)
        tier_state = self._state.tiers.get(chosen_tier)
        round_clients = round_clients_by_tier.get(chosen_tier, [])
        if tier_state is None or not round_clients:
            logger.warning(
                "TiFL: selected tier %d has no available clients this round",
                chosen_tier,
            )
            self._state.last_selected_tier = None
            return [], []

        k = min(self.num_to_select, len(round_clients))
        selected = self._rng.sample(round_clients, k)
        messages = self._make_train_messages(selected, arrays, round_num)

        # Update states: tier credits and last selected tier
        tier_state.credits = max(0, tier_state.credits - 1)
        self._state.last_selected_tier = chosen_tier

        return selected, messages

    def configure_post_training_round(self, round_num, replies, selected_pids):
        super().configure_post_training_round(round_num, replies, selected_pids)

        # Recompute accuracy proxy for ALL tiers from ClientState.last_train_loss.
        # super() has already updated client_states from this round's feedback.
        # accuracy = -mean_loss: higher (less negative) = lower loss = well represented.
        # Clients with no loss yet (last_train_loss is None) are excluded from the mean.
        for tier in self._state.tiers.values():
            known_losses = [
                self.client_states[pid].last_train_loss
                for pid in tier.clients
                if self.client_states[pid].last_train_loss is not None
            ]
            if known_losses:
                tier.accuracy = -sum(known_losses) / len(known_losses)

    def log_metrics(self, round_num, replies, selected_pids, extra_metrics=None):
        tifl_extras = {
            "strategy/tifl_selected_tier": self._state.last_selected_tier,
        }

        tier_probs = [t.probability for t in self._state.tiers.values()]
        if len(tier_probs) > 1:
            import math

            entropy = -sum(p * math.log(p + 1e-9) for p in tier_probs)
            tifl_extras["strategy/tifl_tier_prob_entropy"] = entropy

        for tier_idx, tier in self._state.tiers.items():
            tifl_extras[f"strategy/tifl_tier{tier_idx}/prob"] = float(tier.probability)
            tifl_extras[f"strategy/tifl_tier{tier_idx}/credits"] = float(tier.credits)
            tifl_extras[f"strategy/tifl_tier{tier_idx}/accuracy"] = float(tier.accuracy)

        if extra_metrics:
            tifl_extras.update(extra_metrics)

        return super().log_metrics(round_num, replies, selected_pids, tifl_extras)

    def _build_tiers(self, latency_by_pid: dict[int, float]) -> None:
        sorted_items = sorted(latency_by_pid.items(), key=lambda x: x[1])
        pids_sorted = [pid for pid, _ in sorted_items]

        self._state.tiers = {}
        chunks = np.array_split(np.array(pids_sorted, dtype=int), self.num_tiers)

        for tier_idx, chunk in enumerate(chunks):
            if chunk.size == 0:
                continue
            clients = [int(pid) for pid in chunk.tolist()]
            self._state.tiers[tier_idx] = TiFLTierState(
                tier_id=tier_idx,
                clients=clients,
                client_latencies={pid: latency_by_pid[pid] for pid in clients},
            )

        self.num_tiers = max(1, len(self._state.tiers))

    def _initialize_tier_controls(self) -> None:
        active_tiers = list(self._state.tiers.values())
        if not active_tiers:
            return

        credits_per_tier = max(
            1, math.ceil(self._state.total_rounds / max(1, len(active_tiers)))
        )
        uniform_prob = 1.0 / len(active_tiers)
        for tier in active_tiers:
            tier.credits = credits_per_tier
            tier.probability = uniform_prob
            tier.accuracy = 0.0
            tier.last_checked_accuracy = 0.0

    def _log_tier_structure(self) -> None:
        """Log tier composition and latency statistics once after tier construction."""
        if not self.use_wandb:
            return

        import statistics
        import wandb

        tier_table = wandb.Table(
            columns=[
                "tier_id",
                "n_clients",
                "latency_mean",
                "latency_std",
                "latency_min",
                "latency_max",
                "credits",
                "probability",
            ]
        )

        for tier_idx, tier in self._state.tiers.items():
            latencies = tier.client_latencies.values()
            tier_table.add_data(
                tier_idx,
                len(tier.clients),
                round(statistics.mean(latencies), 3),
                round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 3),
                round(min(latencies), 3),
                round(max(latencies), 3),
                tier.credits,
                round(tier.probability, 4),
            )

        tifl_one_time_metadata = {
            "tifl/tier_structure": tier_table,
            "tifl/active_clients": float(len(self._state.active_clients)),
            "tifl/dropouts": float(len(self._state.dropout_clients)),
        }
        wandb.log(tifl_one_time_metadata, commit=False)

    def _weighted_choose_tier(self, available_tiers: list[int]) -> int:
        probs = [
            max(0.0, self._state.tiers.get(t).probability) for t in available_tiers
        ]
        sum_probs = sum(probs)

        if sum_probs <= 0:
            return self._rng.choice(available_tiers)

        probs = [p / sum_probs for p in probs]
        return self._rng.choices(available_tiers, weights=probs, k=1)[0]

    def _maybe_update_probs(self) -> None:
        if self._state.last_selected_tier is None:
            return
        tier_state = self._state.tiers.get(self._state.last_selected_tier)
        if tier_state is None:
            return

        if tier_state.accuracy <= tier_state.last_checked_accuracy:
            self._change_probs()
            
        for tier in self._state.tiers.values():
            tier.last_checked_accuracy = tier.accuracy

    def _change_probs(self) -> None:
        eligible_tiers = [
            tier
            for tier in self._state.tiers.values()
            if tier.credits > 0 and tier.clients
        ]

        # Set probability to 0 for ineligible tiers (no credits or no clients) and exclude them from the ranking and probability update.
        non_eligible_tiers = [
            tier for tier in self._state.tiers.values() if tier not in eligible_tiers
        ]
        for tier in non_eligible_tiers:
            tier.probability = 0.0

        n = len(eligible_tiers)
        if n <= 1:
            return

        ranked_by_acc = sorted(eligible_tiers, key=lambda tier: tier.accuracy)
        d = n * (n - 1) / 2.0
        if d <= 0:
            return

        new_probs: dict[int, float] = {}
        for rank, tier in enumerate(ranked_by_acc, start=1):
            new_probs[tier.tier_id] = (n - rank) / d

        total = sum(new_probs.values())

        if total <= 0:
            uniform = 1.0 / n
            for tier in eligible_tiers:
                tier.probability = uniform
            return

        for tier in eligible_tiers:
            tier.probability = new_probs.get(tier.tier_id, 0.0) / total

    def _strategy_name(self) -> str:
        return (
            f"TiFL(K={self.num_to_select},tiers={self.num_tiers},"
            f"sync={self.sync_rounds})"
        )
