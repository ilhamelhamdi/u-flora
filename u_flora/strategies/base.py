"""BaseStrategy: base class for all client selection strategies."""

from __future__ import annotations

import io
import logging
import statistics
import time
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from logging import INFO
from typing import Any

import wandb
from flwr.app import MessageType
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid
from flwr.serverapp.strategy import Result, Strategy, strategy_utils

from ..selection.base import (
    ClientState,
    compute_gini_coefficient,
    compute_jain_fairness_index,
)

logger = logging.getLogger(__name__)


class BaseStrategy(Strategy):
    """Abstract base for client selection strategies.

    Args:
        client_states: Registry of per-client mutable state, keyed by partition_id.
            All profiles start as None; each strategy populates them at runtime.
        save_path: Directory passed to clients for saving model checkpoints.
        use_wandb: Whether to emit per-round and summary metrics to W&B.
    """

    # Keys used in RecordDict payloads (must match client_app.py)
    arrayrecord_key: str = "arrays"
    configrecord_key: str = "config"
    weighted_by_key: str = "num-examples"

    def __init__(
        self,
        client_states: dict[int, ClientState],
        save_path: str = "",
        use_wandb: bool = True,
        **kwargs: Any,
    ) -> None:
        self.client_states = client_states
        self.save_path = save_path
        self.use_wandb = use_wandb

        # Built at start() by querying all nodes
        self._pid_to_nid: dict[int, int] = {}  # partition_id → node_id
        self._nid_to_pid: dict[int, int] = {}  # node_id → partition_id

        # Cumulative wall-clock time (sum of per-round max durations)
        self._cumulative_wall_clock: float = 0.0

        # Best centralised evaluation accuracy seen so far
        self._best_accuracy: float = 0.0
        self._best_accuracy_round: int = 0

        # How many times each partition_id has been selected (for fairness metrics)
        self._participation_counts: dict[int, int] = defaultdict(int)

        if use_wandb:
            self._history_table = wandb.Table(
                columns=[
                    "round",
                    "node_id",
                    "train_loss",
                    "duration",
                    "compute_time",
                    "upload_time",
                    "download_time",
                    "num_samples",
                    "selected_by",
                ],
                log_mode="INCREMENTAL",
            )

    # ------------------------------------------------------------------
    # Main entry point (overrides Strategy.start)
    # ------------------------------------------------------------------

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600.0,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the full federated learning run.

        Args:
            grid: Flower Grid for communicating with client nodes.
            initial_arrays: Starting global model parameters.
            num_rounds: Number of federated rounds to run.
            timeout: Per-round communication timeout in seconds.
            train_config: Unused; kept for Strategy ABC compatibility.
            evaluate_config: Unused; kept for Strategy ABC compatibility.
            evaluate_fn: Optional centralized evaluation function called each round.
        """
        logger.info("Starting %s", self.__class__.__name__)
        strategy_utils.log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )

        result = Result()
        t_start = time.time()

        # Step 1: node_id <-> partition_id mapping
        logger.info("Discovering nodes...")
        self.build_node_mapping(grid)

        # Step 2: Pre-training phase for initial client profiling. Default: no-op.
        arrays = initial_arrays
        logger.info("Pre-training phase...")
        pretrain_arrays = self.configure_pretrain(grid, arrays, timeout)
        if pretrain_arrays is not None:
            arrays = pretrain_arrays
            logger.info("Pre-training phase complete")
        else:
            logger.debug("Pre-training phase: no-op")

        # Step 3: Centralized evaluation before training
        if evaluate_fn:
            res = evaluate_fn(0, arrays)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        # Step 4: Main training loop
        logger.info("Starting federated training (%d rounds)...", num_rounds)
        for current_round in range(1, num_rounds + 1):
            logger.info("")
            logger.info("[ROUND %d/%d]", current_round, num_rounds)

            # configure_train: selection + build messages
            selected_pids, messages = self.configure_train(
                current_round, arrays, grid, timeout
            )
            if not messages:
                logger.warning(
                    "Round %d: no messages produced, skipping", current_round
                )
                continue

            logger.info(
                "[ROUND %d/%d] Selected %d clients, sending train messages...",
                current_round, num_rounds, len(selected_pids),
            )

            # Send messages and collect replies
            replies = grid.send_and_receive(messages, timeout=timeout)
            logger.info(
                "[ROUND %d/%d] Received %d/%d replies",
                current_round, num_rounds, len(replies), len(messages),
            )

            # aggregate_train: FedAvg + metrics logging
            new_arrays, train_metrics = self.aggregate_train(
                current_round, replies, selected_pids
            )
            if new_arrays is not None:
                arrays = new_arrays
            if train_metrics is not None:
                result.train_metrics_clientapp[current_round] = train_metrics

            # Centralized evaluation
            if evaluate_fn:
                res = evaluate_fn(current_round, arrays)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._update_best_accuracy(current_round, res)

        result.arrays = arrays
        logger.info("")
        logger.info("Finished in %.2fs", time.time() - t_start)
        logger.info("")
        for line in io.StringIO(str(result)):
            logger.info("\t%s", line.strip("\n"))

        self._log_summary(num_rounds, result)
        return result

    # ------------------------------------------------------------------
    # Node mapping
    # ------------------------------------------------------------------

    def build_node_mapping(self, grid: Grid) -> None:
        """Identify all nodes and build bidirectional partition_id <-> node_id map."""
        all_node_ids = list(grid.get_node_ids())
        record = RecordDict({self.configrecord_key: ConfigRecord()})
        message_type = f"{MessageType.QUERY}.identify"
        messages = self._construct_messages(record, all_node_ids, message_type)
        replies = grid.send_and_receive(messages)

        for reply in replies:
            if reply.has_error():
                logger.warning("Node %s failed to identify", reply.metadata.src_node_id)
                continue
            nid = reply.metadata.src_node_id
            pid = int(reply.content[self.configrecord_key]["partition_id"])
            self._pid_to_nid[pid] = nid
            self._nid_to_pid[nid] = pid

        logger.info(
            "Node mapping built: %d/%d nodes identified",
            len(self._pid_to_nid),
            len(all_node_ids),
        )

    # ------------------------------------------------------------------
    # Strategy hooks — override in subclasses
    # ------------------------------------------------------------------

    def configure_pretrain(
        self,
        grid: Grid,
        arrays: ArrayRecord,
        timeout: float,
    ) -> ArrayRecord | None:
        """Pre-training phase (e.g. TiFL profiling). Default: no-op."""
        return None

    @abstractmethod
    def configure_train(
        self,
        round_num: int,
        arrays: ArrayRecord,
        grid: Grid,
        timeout: float,
    ) -> tuple[list[int], list[Message]]:
        """Select clients and build training messages for one round.

        Returns:
            (selected_pids, messages) — partition_ids of selected clients and
            the corresponding train Message objects to send.
        """
        ...

    def aggregate_train(
        self,
        round_num: int,
        replies: list[Message],
        selected_pids: list[int],
        extra_metrics: dict[str, float] | None = None,
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """FedAvg aggregation over training replies + comprehensive metrics logging.

        Args:
            round_num: Current round number.
            replies: Raw reply messages from clients.
            selected_pids: Partition IDs that were asked to train this round.
            extra_metrics: Strategy-specific floats added to W&B log
                           (e.g. ``{"strategy/pacer_T": 120.0}``).

        Returns:
            (aggregated_arrays, aggregated_metric_record)
        """
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            logger.warning(
                "Round %d: no valid replies, skipping aggregation", round_num
            )
            return None, None

        # FedAvg aggregation
        # TODO: Need to be adjusted with FedExLoRA
        records = [msg.content for msg in valid_replies]
        agg_arrays = strategy_utils.aggregate_arrayrecords(
            records, self.weighted_by_key
        )

        # Build aggregated MetricRecord (aggregate numeric metrics)
        agg_metrics = strategy_utils.aggregate_metricrecords(
            [msg.content for msg in valid_replies],
            weighting_metric_name=self.weighted_by_key,
        )

        to_return = agg_arrays, agg_metrics

        # === Additional step: update client states and log metrics ===

        # Extract per-client feedback and update client states
        feedbacks = self.extract_feedback(valid_replies)
        self._update_client_states(round_num, feedbacks, selected_pids)

        # Log metrics
        strategy_name = self._strategy_name()
        self._log_round_metrics(
            round_num, feedbacks, selected_pids, strategy_name, extra_metrics
        )

        return to_return

    # ------------------------------------------------------------------
    # Feedback extraction
    # ------------------------------------------------------------------

    def extract_feedback(
        self, valid_replies: list[Message]
    ) -> dict[int, dict[str, float]]:
        """Extract per-client metrics from valid training replies.

        Returns:
            Mapping from partition_id → {train_loss, duration, compute_time,
            upload_time, download_time, num_samples}.
        """
        feedbacks: dict[int, dict[str, float]] = {}
        for msg in valid_replies:
            m = (
                msg.content["metrics"]
                if "metrics" in msg.content
                else msg.content[self.weighted_by_key]
            )
            pid = int(m["node_id"])  # client reports its partition_id as "node_id"
            feedbacks[pid] = {
                "train_loss": float(m.get("train_loss", 0.0)),
                "duration": float(m.get("duration", 0.0)),
                "compute_time": float(m.get("actual_train_time", 0.0)),
                "upload_time": 0.0,  # not yet tracked separately by client
                "download_time": 0.0,  # not yet tracked separately by client
                "num_samples": float(m.get("num-examples", 0)),
            }
        return feedbacks

    # ------------------------------------------------------------------
    # Strategy ABC stubs (centralized eval not used)
    # ------------------------------------------------------------------

    def configure_evaluate(self, server_round, arrays, grid, timeout, config=None):
        return [], []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def summary(self):
        pass

    # ------------------------------------------------------------------
    # Message helpers (copied from FedAvg)
    # ------------------------------------------------------------------

    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: str
    ) -> Iterable[Message]:
        """Construct N Messages carrying the same RecordDict payload."""
        messages = []
        for node_id in node_ids:
            message = Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages

    def _check_and_log_replies(
        self, replies: Iterable[Message], is_train: bool, validate: bool = True
    ) -> tuple[list[Message], list[Message]]:
        """Check replies for errors and log them (copied from FedAvg)."""
        if not replies:
            return [], []

        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(
            INFO,
            "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate",
            len(valid_replies),
            len(error_replies),
        )

        for msg in error_replies:
            log(
                INFO,
                "\t> Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error.reason,
            )

        if validate and valid_replies:
            strategy_utils.validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=is_train,
            )

        return valid_replies, error_replies

    def _make_train_messages(
        self,
        partition_ids: list[int],
        arrays: ArrayRecord,
        server_round: int,
    ) -> list[Message]:
        """Build train messages for the given partition IDs."""
        record = RecordDict(
            {
                self.arrayrecord_key: arrays,
                self.configrecord_key: ConfigRecord(
                    {
                        "server-round": server_round,
                        "save_path": self.save_path,
                    }
                ),
            }
        )
        node_ids = [self._pid_to_nid[p] for p in partition_ids if p in self._pid_to_nid]
        return list(self._construct_messages(record, node_ids, MessageType.TRAIN))

    def _make_query_messages(
        self,
        partition_ids: list[int],
        action: str,
        extra_config: dict | None = None,
    ) -> list[Message]:
        """Build query messages for the given partition IDs."""
        cfg_dict: dict = {}
        if extra_config:
            cfg_dict.update(extra_config)
        message_type = f"{MessageType.QUERY}.{action}"
        record = RecordDict({self.configrecord_key: ConfigRecord(cfg_dict)})
        node_ids = [self._pid_to_nid[p] for p in partition_ids if p in self._pid_to_nid]
        return list(self._construct_messages(record, node_ids, message_type))

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _update_client_states(
        self,
        round_num: int,
        feedbacks: dict[int, dict[str, float]],
        selected_pids: list[int],
    ) -> None:
        """Update ClientState from training feedback and participation counts."""
        for pid, fb in feedbacks.items():
            if pid in self.client_states:
                self.client_states[pid].update_from_feedback(
                    train_loss=fb.get("train_loss", 0.0),
                    duration_s=fb.get("duration", 0.0),
                    current_round=round_num,
                )
                self._participation_counts[pid] += 1

    def _update_best_accuracy(self, round_num: int, metrics: MetricRecord) -> None:
        """Track best centralised accuracy seen so far."""
        acc = metrics.get("eval_accuracy") or metrics.get("eval_perplexity")
        if acc is not None:
            # For accuracy higher is better; for perplexity lower is better.
            # Store as-is and let _log_summary decide.
            val = float(acc)
            if val > self._best_accuracy:
                self._best_accuracy = val
                self._best_accuracy_round = round_num

    def _strategy_name(self) -> str:
        """Human-readable name for W&B logging. Override in subclasses."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Metrics logging
    # ------------------------------------------------------------------

    def _log_round_metrics(
        self,
        round_num: int,
        feedbacks: dict[int, dict[str, float]],
        selected_pids: list[int],
        strategy_name: str,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log per-round and per-client metrics to W&B."""
        if not feedbacks:
            return

        durations = [fb["duration"] for fb in feedbacks.values()]
        wall_clock = max(durations)
        self._cumulative_wall_clock += wall_clock

        dur_mean = statistics.mean(durations)
        dur_std = statistics.stdev(durations) if len(durations) > 1 else 0.0

        # Fairness over full participation history
        all_counts = [self._participation_counts[pid] for pid in self.client_states]
        jfi = compute_jain_fairness_index(all_counts)
        gini = compute_gini_coefficient(all_counts)
        p_std = statistics.stdev(all_counts) if len(all_counts) > 1 else 0.0

        # Log per-client details
        if self.use_wandb:
            for pid, fb in feedbacks.items():
                self._history_table.add_data(
                    round_num,
                    pid,
                    fb["train_loss"],
                    fb["duration"],
                    fb["compute_time"],
                    fb["upload_time"],
                    fb["download_time"],
                    int(fb["num_samples"]),
                    strategy_name,
                )

            log_dict: dict[str, float] = {
                "round/server_round": round_num,
                "round/wall_clock": wall_clock,
                "round/cumulative_wall_clock": self._cumulative_wall_clock,
                "round/duration_mean": dur_mean,
                "round/duration_std": dur_std,
                "round/num_selected": len(selected_pids),
                "round/num_completed": len(feedbacks),
                "fairness/jain_index": jfi,
                "fairness/participation_std": p_std,
                "fairness/participation_min": float(min(all_counts)),
                "fairness/participation_max": float(max(all_counts)),
            }
            if extra_metrics:
                log_dict.update(extra_metrics)

            wandb.log(log_dict, commit=False)
            wandb.log(
                {"client/raw_training_history": self._history_table}, commit=False
            )

        logger.info(
            "Round %d — wall_clock=%.1fs cumul=%.1fs dur_mean=%.1f±%.1f "
            "selected=%d completed=%d JFI=%.3f",
            round_num,
            wall_clock,
            self._cumulative_wall_clock,
            dur_mean,
            dur_std,
            len(selected_pids),
            len(feedbacks),
            jfi,
        )

    def _log_summary(self, num_rounds: int, result: Result) -> None:
        """Log end-of-training summary metrics to W&B."""
        all_counts = [self._participation_counts[pid] for pid in self.client_states]
        jfi = compute_jain_fairness_index(all_counts)
        gini = compute_gini_coefficient(all_counts)

        # Retrieve last eval metrics if available
        last_round_with_eval = max(result.evaluate_metrics_serverapp, default=None)
        final_accuracy: float | None = None
        if last_round_with_eval is not None:
            m = result.evaluate_metrics_serverapp[last_round_with_eval]
            final_accuracy = float(
                m.get("eval_accuracy", m.get("eval_perplexity", 0.0))
            )

        if self.use_wandb:
            summary: dict[str, float] = {
                "summary/total_wall_clock": self._cumulative_wall_clock,
                "summary/final_jain_index": jfi,
                "summary/participation_gini": gini,
                "summary/best_accuracy_round": float(self._best_accuracy_round),
                "summary/best_accuracy": self._best_accuracy,
            }
            if final_accuracy is not None:
                summary["summary/final_accuracy"] = final_accuracy
            wandb.log(summary)

        logger.info(
            "Summary — total_wall_clock=%.1fs JFI=%.3f Gini=%.3f best_acc=%.4f (round %d)",
            self._cumulative_wall_clock,
            jfi,
            gini,
            self._best_accuracy,
            self._best_accuracy_round,
        )
