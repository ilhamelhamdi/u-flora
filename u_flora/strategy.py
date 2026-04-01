"""Custom Flower strategy with pluggable client selection.

Extends FedAvg to use the strategy-pattern ``ClientSelector`` interface.
The selection algorithm is chosen via Hydra config (strategy/*.yaml) and
injected at construction time.

Lifecycle per round:
  1. ``configure_train()`` → call ``selector.select_clients()`` to choose participants
  2. Training happens on selected clients (Flower handles this)
  3. ``aggregate_train()`` → log metrics, call ``selector.update_feedback()``
  4. ``evaluate()`` → centralized evaluation on validation set
"""

from __future__ import annotations

import logging
from typing import Any

import wandb
from flwr.serverapp.strategy import FedAvg

from .selection.base import ClientSelector, ClientState

logger = logging.getLogger(__name__)


class SelectionStrategy(FedAvg):
    """FedAvg aggregation + pluggable client selection.

    This replaces Flower's random fraction-based selection with our
    selector implementations (Random, FedCS, Oort, etc.).

    Args:
        selector: A ``ClientSelector`` instance that picks clients each round.
        client_states: Pre-built registry mapping client_id → ClientState.
            Populated from device profiles at experiment start.
        fraction_train: Fallback fraction if selector is None.
        fraction_evaluate: Fraction of clients for evaluation (0 = skip).
        use_wandb: Whether to log metrics to Weights & Biases.
    """

    def __init__(
        self,
        selector: ClientSelector | None = None,
        client_states: dict[int, ClientState] | None = None,
        fraction_train: float = 0.2,
        fraction_evaluate: float = 0.0,
        use_wandb: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )
        self.selector = selector
        self.client_states = client_states or {}
        self.use_wandb = use_wandb

        if self.use_wandb:
            self._history_table = wandb.Table(
                columns=[
                    "round", "node_id", "duration",
                    "train_loss", "selected_by",
                ],
                log_mode="INCREMENTAL",
            )

    # ------------------------------------------------------------------
    # Selection override
    # ------------------------------------------------------------------

    def _select_client_ids(self, server_round: int) -> list[int] | None:
        """Use the plugged selector instead of Flower's random selection.

        Returns None to fall back to Flower's default if no selector.
        """
        if self.selector is None or not self.client_states:
            return None

        selected = self.selector.select_clients(
            server_round, self.client_states
        )
        logger.info(
            "Round %d: %s selected %d clients: %s",
            server_round,
            self.selector.name,
            len(selected),
            selected[:5],
        )
        return selected

    # ------------------------------------------------------------------
    # Aggregation with feedback tracking
    # ------------------------------------------------------------------

    def aggregate_train(self, server_round: int, replies):
        """Aggregate training results and update selector feedback."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        round_durations: list[float] = []
        feedbacks: dict[int, dict[str, float]] = {}

        for msg in valid_replies:
            metrics_key = (
                self.weighted_by_key
                if "metrics" not in msg.content
                else "metrics"
            )
            metrics = msg.content[metrics_key]

            node_id = int(metrics["node_id"])
            duration = float(metrics["duration"])
            loss = float(metrics["train_loss"])

            round_durations.append(duration)
            feedbacks[node_id] = {
                "train_loss": loss,
                "duration": duration,
            }

            # W&B logging
            if self.use_wandb:
                selector_name = self.selector.name if self.selector else "FedAvg"
                self._history_table.add_data(
                    server_round, node_id, duration, loss, selector_name
                )
                wandb.log(
                    {
                        f"client/node_{node_id}/duration": duration,
                        f"client/node_{node_id}/train_loss": loss,
                        "round/server_round": server_round,
                    },
                    commit=False,
                )

        # Update selector with feedback
        if self.selector and feedbacks:
            self.selector.update_feedback(
                server_round, feedbacks, self.client_states
            )

        # Log round-level stats
        if round_durations and self.use_wandb:
            wandb.log(
                {
                    "round/avg_duration": sum(round_durations) / len(round_durations),
                    "round/min_duration": min(round_durations),
                    "round/max_duration": max(round_durations),
                    "round/num_completed": len(round_durations),
                },
                commit=False,
            )
            wandb.log(
                {"client/raw_training_history": self._history_table},
                commit=False,
            )

        return super().aggregate_train(server_round, replies)

    @property
    def strategy_name(self) -> str:
        if self.selector:
            return self.selector.name
        return "FedAvg"
