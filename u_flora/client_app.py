"""Flower ClientApp — task-agnostic via TaskAdapter with heterogeneity simulation."""

import json
import logging
import os
import time
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, Trainer

from .tasks.registry import get_task_adapter
from .dataset import load_data
from .utils import replace_keys, cosine_annealing

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

app = ClientApp()


def _load_device_profile() -> dict | None:
    profile_path = os.environ.get("DEVICE_PROFILE_PATH")
    if not profile_path or not os.path.exists(profile_path):
        return None
    with open(profile_path) as f:
        return json.load(f)


def _inject_compute_delay(
    profile: dict | None,
    num_samples: int,
    local_epochs: int,
    actual_train_time_s: float,
) -> float:
    """Sleep to simulate heterogeneous compute capability.

    The device profile's ``computation_latency_ms`` represents per-sample
    training time on the simulated device. The total simulated time is:
        simulated = computation_latency_ms x num_samples x local_epochs / 1000

    We then sleep for (simulated - actual_gpu_time) if positive.

    Returns the total duration including the injected delay.
    """
    if profile is None:
        return actual_train_time_s

    comp_ms = profile.get("computation_latency_ms", 0)
    if comp_ms <= 0:
        return actual_train_time_s

    simulated_train_s = comp_ms * num_samples * local_epochs / 1000.0
    extra_delay = max(0.0, simulated_train_s - actual_train_time_s)

    if extra_delay > 0:
        logger.debug(
            "Injecting %.2fs compute delay (simulated=%.2fs, actual=%.2fs)",
            extra_delay, simulated_train_s, actual_train_time_s,
        )
        time.sleep(extra_delay)

    return actual_train_time_s + extra_delay


@app.train()
def train(msg: Message, context: Context):
    # -- Parse config --------------------------------------------------
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    num_rounds = cfg.num_server_rounds
    server_round = msg.content["config"]["server-round"]

    # Resolve task & dataset
    task_name = getattr(cfg, "task", "text_classification")
    chosen_dataset = cfg.dataset
    dataset_config = cfg.datasets[chosen_dataset]

    # -- Task adapter --------------------------------------------------
    adapter = get_task_adapter(task_name)

    # -- Data ----------------------------------------------------------
    encoding_func = adapter.get_encoding_fn(cfg.model.name, dataset_config)
    data_collator = adapter.get_data_collator(cfg.model.name)

    train_set, _ = load_data(partition_id, num_partitions, dataset_config)
    train_set = train_set.map(encoding_func, batched=True)

    # -- Model ---------------------------------------------------------
    model = adapter.get_model(cfg.model)
    set_peft_model_state_dict(
        model, msg.content["arrays"].to_torch_state_dict()
    )

    # -- Training arguments --------------------------------------------
    task_cfg = getattr(cfg, "task_config", cfg.get("train", {}))
    train_args_dict = dict(
        getattr(task_cfg, "training_arguments", cfg.train.training_arguments)
    )
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
    start_time = time.time()
    results = trainer.train()
    actual_train_time = time.time() - start_time

    # -- Inject compute heterogeneity delay ----------------------------
    profile = _load_device_profile()
    local_epochs = training_arguments.num_train_epochs
    total_duration = _inject_compute_delay(
        profile, len(train_set), int(local_epochs), actual_train_time
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
    return Message(content=content, reply_to=msg)
