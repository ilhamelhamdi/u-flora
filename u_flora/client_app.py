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
import torch
from transformers import Trainer, TrainingArguments

from .tasks.registry import get_task_adapter
from .dataset import load_data
from .utils import replace_keys, cosine_annealing, warmup_then_cosine, FedProxTrainer, configure_logging
from .utils.availability import is_available
from .utils.timing import apply_duration_jitter, estimate_round_duration_s

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    evaluate_during_training = msg.content["config"].get(
        "evaluate_during_training", False
    )

    # Shorthand prefix for all log lines from this client this round
    tag = f"[C{partition_id:03d} | R{server_round}/{num_rounds}]"

    # Resolve task & dataset
    task_name = cfg.task_name
    dataset_name = cfg.dataset.name
    dataset_config = cfg.datasets[dataset_name]
    if "num_labels" in dataset_config:
        cfg.model.num_labels = int(dataset_config.num_labels)

    # -- Task adapter --------------------------------------------------
    adapter = get_task_adapter(task_name)

    # -- Data ----------------------------------------------------------
    logger.debug(
        "%s Loading dataset partition %d/%d...", tag, partition_id, num_partitions
    )
    encoding_func = adapter.get_encoding_fn(cfg.model.name, dataset_config)
    data_collator = adapter.get_data_collator(cfg.model.name)

    train_set, val_set = load_data(partition_id, num_partitions, cfg)
    train_set = train_set.map(encoding_func, batched=True)
    val_set = val_set.map(encoding_func, batched=True)
    logger.debug("%s Dataset ready: %d samples", tag, len(train_set))

    # -- Model ---------------------------------------------------------
    logger.debug("%s Loading model weights...", tag)
    model = adapter.get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    # Capture global params before any local updates (used by FedProx).
    global_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    # -- Training arguments --------------------------------------------
    train_args_dict = dict(cfg.train.training_arguments)
    training_arguments = TrainingArguments(**train_args_dict)

    # Learning rate schedule: linear warmup then cosine annealing.
    warmup_rounds = int(getattr(cfg.train, "warmup_rounds", 0))
    new_lr = warmup_then_cosine(
        server_round,
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
        warmup_rounds=warmup_rounds,
    )
    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]
    training_arguments.report_to = "none"

    # FedProx: use proximal regularization when mu > 0.
    fedprox_mu = float(
        getattr(getattr(cfg.train, "fedprox", None), "mu", 0.0) or 0.0
    )
    trainer_kwargs: dict = dict(
        model=model,
        args=training_arguments,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        compute_metrics=adapter.compute_metrics,
    )
    if fedprox_mu > 0.0:
        trainer_kwargs["global_params"] = global_params
        trainer_kwargs["mu"] = fedprox_mu
        trainer = FedProxTrainer(**trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    try:
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

        # -- Evaluation --------------------------------------------------
        # Evaluation phase is only needed for some strategies (TiFL).
        if evaluate_during_training:
            logger.info("%s Validation started — samples=%d", tag, len(val_set))
            eval_metrics = trainer.evaluate()
            metric_name = adapter.get_metric_name()
            local_val_metric = eval_metrics.get(f"eval_{metric_name}", float("nan"))
            local_val_loss = eval_metrics.get("eval_loss", float("nan"))
            logger.info(
                "%s Validation done — loss=%.4f  %s=%.4f",
                tag,
                local_val_loss,
                metric_name,
                local_val_metric,
            )

        # -- Compute simulated wall-clock duration -------------------------
        profile = _load_device_profile(context)
        local_epochs = int(training_arguments.num_train_epochs)
        sim_compute_s, sim_comm_s, sim_total_s, jitter_meta = (
            _compute_simulated_duration(
                profile=profile,
                num_samples=len(train_set),
                local_epochs=local_epochs,
                model_size_kb=adapter.get_lora_adapter_size_kb(cfg.model),
                context=context,
                server_round=server_round,
                partition_id=partition_id,
            )
        )
        logger.debug(
            "%s Simulated duration: total=%.2fs (compute=%.2fs, comm=%.2fs); "
            "actual_train_time=%.2fs",
            tag,
            sim_total_s,
            sim_compute_s,
            sim_comm_s,
            actual_train_time,
        )

        # -- Build response ------------------------------------------------
        model_record = ArrayRecord(get_peft_model_state_dict(model))

        model.to("cpu")
        del trainer, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        metrics = {
            "train_loss": results.training_loss,
            "num-examples": len(train_set),
            "simulated_duration_s": sim_total_s,
            "simulated_compute_s": sim_compute_s,
            "simulated_comm_s": sim_comm_s,
            "actual_train_time": actual_train_time,
            "partition_id": partition_id,
            **jitter_meta,
        }

        if evaluate_during_training:
            metrics[f"val_{metric_name}"] = local_val_metric
            metrics["val_loss"] = local_val_loss
            metrics["val_num_examples"] = len(val_set)

        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        logger.debug("%s Response ready — simulated_duration=%.1fs", tag, sim_total_s)
        return Message(content=content, reply_to=msg)
    except Exception as e:
        # Cleanup
        if "model" in locals():
            model.to("cpu")
        del trainer
        if "model" in locals():
            del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()


@app.query("identify")
def handle_identify(msg: Message, context: Context) -> Message:
    """Return this node's partition_id so the server can build its mapping."""
    partition_id = int(context.node_config["partition-id"])
    logger.debug("[C%03d] Identify query received", partition_id)
    return Message(
        content=RecordDict({"config": ConfigRecord({"partition_id": partition_id})}),
        reply_to=msg,
    )


@app.query("heartbeat")
def handle_heartbeat(msg: Message, context: Context) -> Message:
    """Reply with availability at the server-supplied virtual time."""
    partition_id = int(context.node_config["partition-id"])
    virtual_time_s = float(msg.content["config"].get("virtual_time_s", 0.0))

    profile = _load_device_profile(context)
    behavior = profile.get("behavior") if isinstance(profile, dict) else None

    if behavior is None:
        available = True
    else:
        available = bool(is_available(behavior, virtual_time_s))

    logger.debug(
        "[C%03d] Heartbeat at t=%.1fs -> available=%s",
        partition_id,
        virtual_time_s,
        available,
    )
    return Message(
        content=RecordDict(
            {
                "config": ConfigRecord(
                    {"available": available, "partition_id": partition_id}
                )
            }
        ),
        reply_to=msg,
    )


@app.query("resource_request")
def handle_resource_request(msg: Message, context: Context) -> Message:
    """Return device capability + dataset size (FedCS / TiFL profiling reply)."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    logger.debug("[C%03d] Resource request received", partition_id)

    # Load dataset to report num_samples
    train_set, _ = load_data(partition_id, num_partitions, cfg)
    num_samples = len(train_set)

    profile = _load_device_profile(context)
    response = {
        "partition_id": partition_id,
        "num_samples": num_samples,
        "computation": float(profile["computation"]) if profile else 0.0,
        "communication": float(profile["communication"]) if profile else 0.0,
    }
    return Message(
        content=RecordDict({"config": ConfigRecord(response)}),
        reply_to=msg,
    )


# -- Profile loading -----------------------------------------------------------


def _resolve_profile_path(context: Context) -> str | None:
    """Find the profile file path."""
    run_path = (
        context.run_config.get("device-profile-path") if context.run_config else None
    )
    if run_path:
        return str(run_path)
    return context.node_config.get("device-profile-path")


def _resolve_profile_key(context: Context) -> str:
    """Find the profile key for this client."""
    explicit = context.node_config.get("device-profile-key")
    if explicit:
        return str(explicit)
    pid = int(context.node_config["partition-id"])
    return f"supernode-{pid:03d}"


def _load_device_profile(context: Context) -> dict | None:
    """Resolve and load the FedScale-shaped device profile for this client."""
    profile_path = _resolve_profile_path(context)
    if not profile_path:
        logger.warning(
            "device-profile-path not set in run_config or node_config — "
            "simulated duration will be 0."
        )
        return None
    if not os.path.exists(profile_path):
        logger.warning(
            "Device profile not found at %s — simulated duration will be 0.",
            profile_path,
        )
        return None

    with open(profile_path) as f:
        profiles = json.load(f)

    key = _resolve_profile_key(context)
    profile = profiles.get(key)
    if profile is None:
        logger.warning(
            "Profile key '%s' not found in %s — simulated duration will be 0.",
            key,
            profile_path,
        )
        return None
    logger.debug("Loaded device profile %s from %s", key, profile_path)
    return profile


def _compute_simulated_duration(
    profile: dict | None,
    num_samples: int,
    local_epochs: int,
    model_size_kb: float,
    context: Context,
    server_round: int,
    partition_id: int,
) -> tuple[float, float, float, dict]:
    if profile is None:
        return 0.0, 0.0, 0.0, {}

    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    compute_s, comm_s, _ = estimate_round_duration_s(
        profile=profile,
        num_samples=num_samples,
        local_epochs=local_epochs,
        model_size_kb=model_size_kb,
    )

    j = (
        cfg.simulation.jitter
        if "simulation" in cfg and "jitter" in cfg.simulation
        else None
    )
    if j is None or not bool(getattr(j, "enabled", False)):
        return compute_s, comm_s, compute_s + comm_s, {}

    return apply_duration_jitter(
        compute_s,
        comm_s,
        compute_cv=float(getattr(j, "compute_cv", 0.0)),
        comm_cv=float(getattr(j, "comm_cv", 0.0)),
        compute_min_factor=float(getattr(j, "compute_min_factor", 0.7)),
        comm_min_factor=float(getattr(j, "comm_min_factor", 0.5)),
        comm_stall_prob=float(getattr(j, "comm_stall_prob", 0.0)),
        comm_stall_factor_min=float(getattr(j, "comm_stall_factor_min", 5.0)),
        comm_stall_factor_max=float(getattr(j, "comm_stall_factor_max", 10.0)),
        seed=(int(cfg.seed), int(server_round), int(partition_id)),
    )
