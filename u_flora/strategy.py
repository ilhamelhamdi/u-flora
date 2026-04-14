"""Custom Flower strategy with pluggable client selection.

Extends FedAvg to use the strategy-pattern ``ClientSelector`` interface.

Lifecycle per round:
  1. ``configure_train()`` → call ``selector.select_clients()`` to choose participants
  2. Training happens on selected clients (Flower handles this)
  3. ``aggregate_train()`` → log metrics, call ``selector.update_feedback()``
  4. ``evaluate()`` → centralized evaluation on validation set
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any
from collections.abc import Callable, Iterable

import wandb
from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, Result, strategy_utils

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


    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        logger.info( "Starting client selection %s strategy:", self.selector.name)
        strategy_utils.log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        logger.info( "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            logger.info( "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            logger.info( "")
            logger.info( "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                logger.info( "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics


            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                logger.info( "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                logger.info( "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        logger.info( "")
        logger.info( "Strategy execution finished in %.2fs", time.time() - t_start)
        logger.info( "")
        logger.info( "Final results:")
        logger.info( "")
        for line in io.StringIO(str(result)):
            logger.info( "\t%s", line.strip("\n"))
        logger.info( "")

        return result


    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        logger.info( "\t├──> Sampling:")
        logger.info(
            "\t│\t├──Fraction: train (%.2f) | evaluate ( %.2f)",
            self.fraction_train,
            self.fraction_evaluate,
        )  # pylint: disable=line-too-long
        logger.info(
            "\t│\t├──Minimum nodes: train (%d) | evaluate (%d)",
            self.min_train_nodes,
            self.min_evaluate_nodes,
        )  # pylint: disable=line-too-long
        logger.info( "\t│\t└──Minimum available nodes: %d", self.min_available_nodes)
        logger.info( "\t└──> Keys in records:")
        logger.info( "\t\t├── Weighted by: '%s'", self.weighted_by_key)
        logger.info( "\t\t├── ArrayRecord key: '%s'", self.arrayrecord_key)
        logger.info( "\t\t└── ConfigRecord key: '%s'", self.configrecord_key)

    # ------------------------------------------------------------------
    # Selection override
    # ------------------------------------------------------------------

    def _select_client_ids(self, server_round: int) -> list[int] | None:
        """Use the plugged selector instead of Flower's random selection.

        Returns None to fall back to Flower's default if no selector.
        """
        if self.selector is None or not self.client_states:
            logger.warning(
                "Skip client ids selection due to missing `selector` or `client_states`"
            )
            return None

        selected = self.selector.select_clients(
            server_round, self.client_states
        )
        logger.info(
            "Round %d: %s selected %d clients: %s",
            server_round,
            self.selector.name,
            len(selected),
            selected,
        )
        return selected

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        logger.info(
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(self, server_round: int, replies):
        """Aggregate training results and update selector feedback."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        round_durations: list[float] = []
        feedbacks: dict[int, dict[str, float]] = {}

        for msg in valid_replies:
            metrics_key = (
                self.weighted_by_key # FedAvg by default uses "num-examples" key to weight
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
