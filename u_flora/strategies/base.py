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

from ..client_profile.typing import ClientState, DeviceProfile
from ..utils.metrics import compute_gini_coefficient, compute_jain_fairness_index
from ..utils.timing import estimate_round_duration_s

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
        metric_name: str = "accuracy",  # accuracy, f1, perplexity
        is_higher_better: bool = True,
        evaluate_during_training: bool = False,
        model_size_kb: float = 0.0,
        heartbeat_timeout_s: float = 30.0,
        round_timeout_s: float = 600.0,
        history_log_every_n: int = 5,
        **kwargs: Any,
    ) -> None:
        self.client_states = client_states
        self.save_path = save_path
        self.use_wandb = use_wandb
        self.metric_name = metric_name
        self.is_higher_better = is_higher_better
        self.evaluate_during_training = evaluate_during_training
        # Used by strategies that estimate per-client durations server-side
        # (FedCS, TiFL, Oort exploration).
        self.model_size_kb = float(model_size_kb)
        # Per-round heartbeat timeout. Heartbeat itself takes 0 *simulated*
        # time and does not advance the virtual clock.
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.round_timeout_s = float(round_timeout_s)
        self.history_log_every_n = max(1, int(history_log_every_n))
        # Last-round availability snapshot (filled by _check_availability).
        self._last_available_pids: set[int] = set()
        self._last_unavailable_pids: set[int] = set()

        # Built at start() by querying all nodes
        self._pid_to_nid: dict[int, int] = {}  # partition_id → node_id
        self._nid_to_pid: dict[int, int] = {}  # node_id → partition_id

        # Cumulative wall-clock time (sum of per-round max durations)
        self._cumulative_wall_clock: float = 0.0

        # Best centralised evaluation metric seen so far
        self._best_metric: float = float("-inf") if is_higher_better else float("inf")
        self._best_metric_round: int = -1

        # How many times each partition_id has been selected (for fairness metrics)
        self._participation_counts: dict[int, int] = defaultdict(int)

        # Set of clients that have ever been selected or explored
        self._explored_clients: set[int] = set()

        if use_wandb:
            self._history_table = wandb.Table(
                columns=[
                    "round",
                    "node_id",
                    "train_loss",
                    "duration",
                    "compute_time",
                    "communication_time",
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
        num_rounds: int,
        timeout: float | None = None,
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
        if timeout is None:
            timeout = self.round_timeout_s
        logger.info(
            "Starting %s — round_timeout=%.0fs", self.__class__.__name__, timeout
        )
        strategy_utils.log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )

        result = Result()
        t_start = time.time()

        # Step 1: node_id <-> partition_id mapping
        logger.info("Discovering nodes...")
        self._build_node_mapping(grid)
        logger.info("Discovered %d nodes", len(self._pid_to_nid))

        # Step 2: Pre-training phase for initial client profiling. Default: no-op.
        arrays = initial_arrays
        logger.info("Pre-training phase...")
        self.configure_pretrain(grid, arrays, timeout)

        # Step 3: Centralized evaluation before training
        if evaluate_fn:
            logger.info("Running initial evaluation before training...")
            res = evaluate_fn(0, arrays)
            if res is None:
                logger.info("Initial evaluation returned None, skipping.")
            else:
                result.evaluate_metrics_serverapp[0] = res
                logger.info("Initial evaluation metrics: %s", res)
                self._log_pretrain_eval(res)

        # Step 4: Main training loop
        logger.info("Starting federated training (%d rounds)...", num_rounds)
        for current_round in range(1, num_rounds + 1):
            logger.info("")
            logger.info("[ROUND %d/%d]", current_round, num_rounds)

            # Heartbeat: filter unavailable clients before selection.
            virtual_time_s = self._cumulative_wall_clock
            self._check_availability(grid, virtual_time_s)
            n_avail = len(self._last_available_pids)
            n_total = len(self._pid_to_nid)
            logger.info(
                "[ROUND %d/%d] Heartbeat at t=%.1fs: %d/%d clients available",
                current_round,
                num_rounds,
                virtual_time_s,
                n_avail,
                n_total,
            )

            if n_avail == 0:
                logger.warning(
                    "[ROUND %d/%d] No available clients — skipping round",
                    current_round,
                    num_rounds,
                )
                continue

            # configure_train: selection + build messages
            selected_pids, messages = self.configure_train(
                current_round, arrays, grid, timeout
            )

            logger.info(
                "[ROUND %d/%d] Selected %d clients, sending train messages...",
                current_round,
                num_rounds,
                len(selected_pids),
            )

            replies = grid.send_and_receive(messages, timeout=timeout)

            logger.info(
                "[ROUND %d/%d] Received %d/%d replies",
                current_round,
                num_rounds,
                len(replies),
                len(messages),
            )

            # aggregate_train: FedAvg
            new_arrays, train_metrics = self.aggregate_train(current_round, replies)
            if new_arrays is not None:
                arrays = new_arrays
            if train_metrics is not None:
                result.train_metrics_clientapp[current_round] = train_metrics

            # Depending on the strategy, we may want to extract client feedback and update states
            self.configure_post_training_round(current_round, replies, selected_pids)

            # Centralized evaluation
            eval_extra: dict[str, float] | None = None
            if evaluate_fn:
                res = evaluate_fn(current_round, arrays)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._update_best_metric(current_round, res)
                    eval_extra = self._eval_metrics_for_wandb(res)

            # Log metrics (eval scalars merged in)
            self.log_metrics(
                current_round, replies, selected_pids, extra_metrics=eval_extra
            )

        result.arrays = arrays
        logger.info("")
        logger.info("Finished in %.2fs", time.time() - t_start)
        logger.info("")
        for line in io.StringIO(str(result)):
            logger.info("\t%s", line.strip("\n"))

        self._log_summary(num_rounds, result)
        return result

    # ------------------------------------------------------------------
    # Strategy hooks — override in subclasses
    # ------------------------------------------------------------------

    def configure_pretrain(
        self,
        grid: Grid,
        arrays: ArrayRecord,
        timeout: float,
    ) -> None:
        """
        Extension point for optional pre-training phase before main federated rounds.
        By default, this is a no-op that returns None.
        Override in subclasses to implement custom pre-training logic, such as initial client profiling or warm-up rounds.
        """
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
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """FedAvg aggregation over training replies.

        Parameters
        ----------
        server_round : int
            The current round of federated learning, starting from 1.
        replies : Iterable[Message]
            Iterable of reply messages received from client nodes after training.
            Each message contains ArrayRecords and MetricRecords that get aggregated.

        Returns
        -------
        tuple[Optional[ArrayRecord], Optional[MetricRecord]]
            A tuple containing:
            - ArrayRecord: Aggregated ArrayRecord, or None if aggregation failed
            - MetricRecord: Aggregated MetricRecord, or None if aggregation failed
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

        return to_return

    def configure_post_training_round(
        self,
        round_num: int,
        replies: list[Message],
        selected_pids: list[int],
    ):
        # Extract per-client feedback and update client states
        feedbacks = self.extract_feedback(replies)
        self._update_client_states(round_num, feedbacks, selected_pids)

    def log_metrics(
        self,
        round_num: int,
        replies: list[Message],
        selected_pids: list[int],
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        """Extension point for strategy-specific metrics logging."""
        feedbacks = self.extract_feedback(replies)
        self._log_base_metrics(
            round_num=round_num,
            feedbacks=feedbacks,
            selected_pids=selected_pids,
            extra_metrics=extra_metrics,
        )

    # ------------------------------------------------------------------
    # Feedback extraction
    # ------------------------------------------------------------------

    def extract_feedback(
        self, valid_replies: list[Message]
    ) -> dict[int, dict[str, float]]:
        """Extract per-client metrics from valid training replies.

        Returns:
            Mapping from partition_id → {train_loss, duration, compute_time,
            communication_time, num_samples}.
        """
        feedbacks: dict[int, dict[str, float]] = {}
        for msg in valid_replies:
            if not msg.has_content():
                logger.warning(
                    "Skipping a message with no content (likely a client crash/OOM)."
                )
                continue

            m = (
                msg.content["metrics"]
                if "metrics" in msg.content
                else msg.content[self.weighted_by_key]
            )
            pid = int(m["partition_id"])
            feedbacks[pid] = {
                "train_loss": m.get("train_loss"),
                "duration": float(m.get("simulated_duration_s", 0.0)),
                "compute_time": float(m.get("simulated_compute_s", 0.0)),
                "communication_time": float(m.get("simulated_comm_s", 0.0)),
                "num_samples": float(m.get("num-examples", 0)),
            }
            if self.evaluate_during_training:
                metric_key = f"eval_{self.metric_name}"
                feedbacks[pid][metric_key] = m.get(metric_key)
                feedbacks[pid]["val_loss"] = m.get("val_loss")
                feedbacks[pid]["val_num_examples"] = m.get("val_num_examples")

        return feedbacks

    # def filter_replies(self, replies: list[Message]) -> list[Message]:
    #     """Filter out late replies that arrived after the round deadline (for strategies that account for stragglers)."""
    #     # By default, no filtering (all replies are considered valid).
    #     return replies

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
    # Node mapping
    # ------------------------------------------------------------------

    def _build_node_mapping(self, grid: Grid) -> None:
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
        """
        Build train messages for the given partition IDs.
        It will map partition IDs to Flower internal node IDs and construct messages with the given ArrayRecord and server round.
        """
        record = RecordDict(
            {
                self.arrayrecord_key: arrays,
                self.configrecord_key: ConfigRecord(
                    {
                        "server-round": server_round,
                        "save_path": self.save_path,
                        "evaluate_during_training": self.evaluate_during_training,
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
    # Availability heartbeat (per-round)
    # ------------------------------------------------------------------

    def _check_availability(self, grid: Grid, virtual_time_s: float) -> set[int]:
        """Send a heartbeat to every node and update ``ClientState.available``.

        Clients that don't reply (timeout / error) are treated as unavailable.
        Heartbeat takes zero *simulated* time and does not advance the virtual
        clock. Returns the set of available partition_ids.
        """
        all_pids = list(self._pid_to_nid.keys())
        if not all_pids:
            self._last_available_pids = set()
            self._last_unavailable_pids = set()
            return set()

        record = RecordDict(
            {
                self.configrecord_key: ConfigRecord(
                    {"virtual_time_s": float(virtual_time_s)}
                )
            }
        )
        message_type = f"{MessageType.QUERY}.heartbeat"
        messages = self._construct_messages(
            record,
            [self._pid_to_nid[p] for p in all_pids],
            message_type,
        )
        replies = grid.send_and_receive(messages, timeout=self.heartbeat_timeout_s)

        available: set[int] = set()
        for reply in replies:
            if reply.has_error():
                continue
            nid = reply.metadata.src_node_id
            pid = self._nid_to_pid.get(nid)
            if pid is None:
                continue
            cfg = reply.content.get(self.configrecord_key)
            if cfg is None:
                continue
            if bool(cfg.get("available", False)):
                available.add(pid)

        # Anyone in client_states but not in `available` is unavailable
        # (covers timeouts and explicit False replies alike).
        for pid, state in self.client_states.items():
            state.available = pid in available

        self._last_available_pids = available
        self._last_unavailable_pids = set(self.client_states.keys()) - available
        return available

    # ------------------------------------------------------------------
    # Duration estimation (server-side; uses calibration constants)
    # ------------------------------------------------------------------

    def _estimate_duration(
        self,
        profile: DeviceProfile,
        num_samples: int = 0,
        local_epochs: int = 0,
    ) -> tuple[float, float, float]:
        """Estimate (compute_s, comm_s, total_s) for one client one round."""
        return estimate_round_duration_s(
            profile={
                "computation": profile.computation,
                "communication": profile.communication,
            },
            num_samples=num_samples,
            local_epochs=local_epochs,
            model_size_kb=self.model_size_kb,
        )

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
        self._explored_clients.update(feedbacks.keys())
        for pid, fb in feedbacks.items():
            if pid not in self.client_states:
                continue
            self.client_states[pid].update_from_feedback(
                train_loss=fb.get("train_loss", 0.0),
                duration_s=fb.get("duration", 0.0),
                current_round=round_num,
            )
            self._participation_counts[pid] += 1

    def _update_best_metric(self, round_num: int, metrics: MetricRecord) -> None:
        """Track best centralised metric seen so far."""
        metric = metrics.get(f"eval_{self.metric_name}")
        if metric is not None:
            # Determine if the current metric is better based on whether higher is better
            is_better = (
                metric > self._best_metric
                if self.is_higher_better
                else metric < self._best_metric
            )
            if is_better:
                self._best_metric = metric
                self._best_metric_round = round_num

    def _strategy_name(self) -> str:
        """Human-readable name for W&B logging. Override in subclasses."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Metrics logging
    # ------------------------------------------------------------------

    def _log_base_metrics(
        self,
        round_num: int,
        feedbacks: dict[int, dict[str, float]],
        selected_pids: list[int],
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
        unique_clients_explored = len(self._explored_clients)
        exploration_ratio = unique_clients_explored / max(1, len(self.client_states))

        # Utility metrics (Client-losses)
        all_known_losses = [
            state.last_train_loss
            for state in self.client_states.values()
            if state.last_train_loss is not None
        ]
        loss_mean = loss_std = loss_cv = n_clients_with_known_loss = 0.0
        if len(all_known_losses) > 1:
            loss_mean = statistics.mean(all_known_losses)
            loss_std = statistics.stdev(all_known_losses)
            loss_cv = loss_std / loss_mean if loss_mean > 0 else 0.0
            n_clients_with_known_loss = len(all_known_losses)

        # Log per-client details
        if self.use_wandb:
            for pid, fb in feedbacks.items():
                self._history_table.add_data(
                    round_num,
                    pid,
                    fb["train_loss"],
                    fb["duration"],
                    fb["compute_time"],
                    fb["communication_time"],
                    int(fb["num_samples"]),
                    self._strategy_name(),
                )

            log_dict: dict[str, float] = {
                "round/server_round": round_num,
                "round/wall_clock": wall_clock,
                "round/cumulative_wall_clock": self._cumulative_wall_clock,
                "round/duration_mean": dur_mean,
                "round/duration_std": dur_std,
                "round/num_client_selected": len(selected_pids),
                "round/num_client_completed": len(feedbacks),
                "round/n_available": float(len(self._last_available_pids)),
                "round/n_unavailable": float(len(self._last_unavailable_pids)),
                "fairness/jain_index": jfi,
                "fairness/gini_coefficient": gini,
                "fairness/participation_std": p_std,
                "fairness/participation_min": float(min(all_counts)),
                "fairness/participation_max": float(max(all_counts)),
                "fairness/unique_clients_explored": unique_clients_explored,
                "fairness/exploration_ratio": exploration_ratio,
                "utility/loss_mean": loss_mean,
                "utility/loss_std": loss_std,
                "utility/loss_cv": loss_cv,
                "utility/n_clients_with_known_loss": n_clients_with_known_loss,
            }
            if extra_metrics:
                log_dict.update(extra_metrics)

            wandb.log(log_dict)
            if round_num % self.history_log_every_n == 0:
                wandb.log({"client/raw_training_history": self._history_table})

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

    @staticmethod
    def _eval_metrics_for_wandb(res: MetricRecord) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in res.items():
            if not isinstance(v, (int, float)):
                continue
            key = f"eval/{k[len('eval_'):]}" if k.startswith("eval_") else f"eval/{k}"
            out[key] = float(v)
        return out

    def _log_pretrain_eval(self, res: MetricRecord) -> None:
        """Log round-0 evaluation on the same x-axes used during the training loop."""
        if not self.use_wandb:
            return
        log_dict: dict[str, float] = {
            "round/server_round": 0,
            "round/cumulative_wall_clock": 0.0,
        }
        log_dict.update(self._eval_metrics_for_wandb(res))
        wandb.log(log_dict)

    def _log_summary(self, num_rounds: int, result: Result) -> None:
        """Log end-of-training summary metrics to W&B."""
        all_counts = [self._participation_counts[pid] for pid in self.client_states]
        jfi = compute_jain_fairness_index(all_counts)
        gini = compute_gini_coefficient(all_counts)

        # Retrieve last eval metrics if available
        last_round_with_eval = max(result.evaluate_metrics_serverapp, default=None)
        final_metric: float | None = None
        if last_round_with_eval is not None:
            m = result.evaluate_metrics_serverapp[last_round_with_eval]
            final_metric = float(m.get(f"eval_{self.metric_name}", 0.0))

        if self.use_wandb:
            wandb.log(
                {"client/raw_training_history": self._history_table}, commit=False
            )
            summary: dict[str, float] = {
                # --- Performance metrics ---
                "summary/total_wall_clock": self._cumulative_wall_clock,
                "summary/total_rounds": num_rounds,
                "summary/best_metric_round": float(self._best_metric_round),
                "summary/best_metric": self._best_metric,
                "summary/final_metric": (
                    final_metric if final_metric is not None else float("nan")
                ),
                "summary/stability_drop": (
                    self._best_metric - final_metric
                    if final_metric is not None
                    else float("nan")
                ),
                # --- Fairness metrics ---
                "summary/selection_count_histogram": wandb.Histogram(all_counts),
                "summary/clients_never_selected": sum(1 for c in all_counts if c == 0),
                "summary/clients_selected_at_least_once": sum(
                    1 for c in all_counts if c > 0
                ),
                "summary/final_jain_index": jfi,
                "summary/participation_gini": gini,
            }
            wandb.log(summary)

        logger.info(
            "Summary — total_wall_clock=%.1fs JFI=%.3f Gini=%.3f best_metric=%.4f (round %d)",
            self._cumulative_wall_clock,
            jfi,
            gini,
            self._best_metric,
            self._best_metric_round,
        )
