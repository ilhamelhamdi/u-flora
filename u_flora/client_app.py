"""Flower ClientApp"""

import json
import logging
import os
import time
import warnings

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, Trainer

from .tasks.registry import get_task_adapter
from .dataset import load_data
from .utils import replace_keys, cosine_annealing, configure_logging

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

configure_logging(level=os.environ.get("U_FLORA_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_file_logging_configured = False


def _ensure_file_logging(log_path: str) -> None:
    global _file_logging_configured
    if _file_logging_configured or not log_path:
        return
    _LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.getLogger("u_flora").addHandler(handler)
    _file_logging_configured = True

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data using the appropriate TaskAdapter."""
    # -- Parse config --------------------------------------------------
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _ensure_file_logging(context.node_config.get("client-log-path", ""))
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    num_rounds = cfg.num_server_rounds
    server_round = msg.content["config"]["server-round"]

    # Shorthand prefix for all log lines from this client this round
    tag = f"[C{partition_id:03d} | R{server_round}/{num_rounds}]"

    # Resolve task & dataset
    task_name = cfg.task_name
    dataset_name = cfg.dataset.name
    dataset_config = cfg.datasets[dataset_name]

    # -- Task adapter --------------------------------------------------
    adapter = get_task_adapter(task_name)

    # -- Data ----------------------------------------------------------
    logger.debug("%s Loading dataset partition %d/%d...", tag, partition_id, num_partitions)
    encoding_func = adapter.get_encoding_fn(cfg.model.name, dataset_config)
    data_collator = adapter.get_data_collator(cfg.model.name)

    train_set, _ = load_data(partition_id, num_partitions, dataset_config)
    train_set = train_set.map(encoding_func, batched=True)
    logger.debug("%s Dataset ready: %d samples", tag, len(train_set))

    # -- Model ---------------------------------------------------------
    logger.debug("%s Loading model weights...", tag)
    model = adapter.get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    # -- Training arguments --------------------------------------------
    train_args_dict = dict(cfg.train.training_arguments)
    training_arguments = TrainingArguments(**train_args_dict)

    # Learning rate schedule
    new_lr = cosine_annealing(
        server_round,
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )
    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]
    training_arguments.report_to = "none"

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_set,
        data_collator=data_collator,
    )

    # -- Train ---------------------------------------------------------
    logger.info(
        "%s Training started — samples=%d  epochs=%g  lr=%.2e",
        tag,
        len(train_set),
        training_arguments.num_train_epochs,
        new_lr,
    )
    start_time = time.time()
    results = trainer.train()
    actual_train_time = time.time() - start_time
    logger.info(
        "%s Training done — loss=%.4f  time=%.1fs",
        tag,
        results.training_loss,
        actual_train_time,
    )

    # -- Inject compute heterogeneity delay ----------------------------
    profile = _load_device_profile(context.node_config)
    local_epochs = training_arguments.num_train_epochs
    total_duration = _inject_compute_delay(
        profile,
        len(train_set),
        int(local_epochs),
        actual_train_time,
        task_name=task_name,
        node_config=context.node_config,
    )
    if total_duration > actual_train_time:
        logger.debug(
            "%s Simulated duration: %.1fs (added %.1fs compute delay)",
            tag,
            total_duration,
            total_duration - actual_train_time,
        )

    # -- Build response ------------------------------------------------
    model_record = ArrayRecord(get_peft_model_state_dict(model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(train_set),
        "duration": total_duration,
        "actual_train_time": actual_train_time,
        "node_id": partition_id,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    logger.debug("%s Response ready — total_duration=%.1fs", tag, total_duration)
    return Message(content=content, reply_to=msg)


@app.query("identify")
def handle_identify(msg: Message, context: Context) -> Message:
    """Return this node's partition_id so the server can build its mapping."""
    partition_id = int(context.node_config["partition-id"])
    logger.debug("[C%03d] Identify query received", partition_id)
    return Message(
        content=RecordDict({"config": ConfigRecord({"partition_id": partition_id})}),
        reply_to=msg,
    )


@app.query("resource_request")
def handle_resource_request(msg: Message, context: Context) -> Message:
    """Return device capabilities + dataset size (FedCS resource request).

    The server starts blind; this handler is how FedCSWorkflow learns about
    each candidate client's profile before running greedy selection.
    """
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    logger.debug("[C%03d] Resource request received", partition_id)

    # Load dataset to report num_samples
    dataset_config = cfg.datasets[cfg.dataset_name]
    train_set, _ = load_data(partition_id, num_partitions, dataset_config)
    num_samples = len(train_set)

    # Load device profile (ground truth for simulation)
    profile = _load_device_profile(context.node_config)
    response = {
        "partition_id": partition_id,
        "num_samples": num_samples,
        "computation_latency_ms": (
            float(profile["computation_latency_ms"]) if profile else 50.0
        ),
        "download_kbps": float(profile["download_kbps"]) if profile else 5000.0,
        "upload_kbps": float(profile["upload_kbps"]) if profile else 3000.0,
        "latency_ms": float(profile["latency_ms"]) if profile else 50.0,
    }
    return Message(
        content=RecordDict({"config": ConfigRecord(response)}),
        reply_to=msg,
    )


def _load_device_profile(node_config: dict) -> dict | None:
    profile_path = node_config.get("device-profile-path")
    if not profile_path:
        logger.warning("device-profile-path not set in node-config — compute delay simulation disabled")
        return None
    if not os.path.exists(profile_path):
        logger.warning(
            "Device profile not found at %s — compute delay simulation disabled",
            profile_path,
        )
        return None
    with open(profile_path) as f:
        profile = json.load(f)
    logger.debug("Loaded device profile from %s", profile_path)
    return profile


def _load_metadata(node_config: dict) -> dict:
    profile_path = node_config.get("device-profile-path", "")
    metadata_path = os.path.join(os.path.dirname(profile_path), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f)
    logger.warning(
        "Profile metadata not found at %s — compute delay calibration disabled",
        metadata_path,
    )
    return {}


def _inject_compute_delay(
    profile: dict | None,
    num_samples: int,
    local_epochs: int,
    actual_train_time_s: float,
    task_name: str,
    node_config: dict | None = None,
) -> float:
    """Sleep to simulate heterogeneous compute capability.

    Returns the total duration (actual_train_time_s + any injected delay).
    """
    from configs.compute_baseline import T_ACTUAL_MS_PER_SAMPLE

    if profile is None:
        return actual_train_time_s

    t_min_ms = _load_metadata(node_config or {}).get("t_min_ms", 0)
    raw_comp_ms = profile.get("computation_latency_ms", 0)
    if raw_comp_ms <= 0 or t_min_ms <= 0:
        return actual_train_time_s

    t_actual_ms = T_ACTUAL_MS_PER_SAMPLE.get(task_name)
    if t_actual_ms is None or t_actual_ms <= 0:
        logger.warning(
            "No baseline for task '%s'. Skipping compute delay injection.", task_name
        )
        return actual_train_time_s

    slowdown_factor = raw_comp_ms / t_min_ms
    simulated_ms_per_sample = t_actual_ms * slowdown_factor
    simulated_train_s = simulated_ms_per_sample * num_samples * local_epochs / 1000.0

    extra_delay = simulated_train_s - actual_train_time_s

    if extra_delay <= 0:
        logger.debug(
            "Actual GPU time (%.2fs) already exceeds simulated time (%.2fs) "
            "for task '%s', client slowdown=%.2fx. No delay injected.",
            actual_train_time_s,
            simulated_train_s,
            task_name,
            slowdown_factor,
        )
        return actual_train_time_s

    logger.debug(
        "Injecting %.2fs compute delay (simulated=%.2fs, actual=%.2fs, "
        "slowdown=%.2fx, task=%s)",
        extra_delay,
        simulated_train_s,
        actual_train_time_s,
        slowdown_factor,
        task_name,
    )
    time.sleep(extra_delay)
    return actual_train_time_s + extra_delay
