"""Flower ServerApp"""

import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
import torch
import wandb
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from omegaconf import DictConfig, OmegaConf
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from .tasks import TaskAdapter
from .tasks.registry import get_task_adapter
from .dataset import load_data_centralized
from .utils import replace_keys, configure_logging, cast_state_dict_for_arrayrecord
from .client_profile.typing import ClientState
from .strategies.factory import build_strategy

# configure_logging(
#     level=os.environ.get("U_FLORA_LOG_LEVEL", "INFO"),
#     log_file="logs/server_app.log",
# )
logger = logging.getLogger(__name__)

app = ServerApp()


# -- Main ----------------------------------------------------------------------


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/training/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    logger.info(
        "Experiment — task=%s  dataset=%s  model=%s  strategy=%s  rounds=%d  clients=%d",
        cfg.task_name,
        cfg.dataset.name,
        cfg.model.name,
        getattr(cfg.strategy, "name", "random"),
        cfg.num_server_rounds,
        cfg.num_clients,
    )

    # Initialize W&B
    wandb_run = _initialize_wandb(cfg)
    logger.debug("W&B run: %s", wandb_run.name)

    # Resolve task adapter
    task_name = cfg.task_name
    task_adapter = get_task_adapter(task_name)
    logger.debug("Task adapter: %s", task_name)

    dataset_name = cfg.dataset.name
    dataset_config = cfg.datasets[dataset_name]
    if "num_labels" in dataset_config:
        cfg.model.num_labels = int(dataset_config.num_labels)

    # Get initial model weights
    logger.info("Initializing model: %s", cfg.model.name)
    init_model = task_adapter.get_model(cfg.model)
    init_state = get_peft_model_state_dict(init_model)
    arrays = ArrayRecord(cast_state_dict_for_arrayrecord(init_state))
    logger.debug("Model initialized — LoRA state dict keys: %d", len(arrays))

    # Prepare validation set
    logger.info("Loading validation set...")
    val_set, data_collator = _get_validation_set(cfg, task_adapter)
    logger.info("Validation set loaded: %d examples", len(val_set))

    # Build empty client state registry — profiles are discovered at runtime
    num_clients = cfg.num_clients
    client_states = _build_client_states(num_clients)
    logger.debug("Client state registry: %d clients", num_clients)

    # Build per-strategy class (selection + Flower integration)
    strategy = build_strategy(
        cfg,
        client_states,
        save_path,
        model_size_kb=task_adapter.get_lora_adapter_size_kb(cfg.model),
        use_wandb=True,
        metric_name=task_adapter.get_metric_name(),
        is_higher_better=task_adapter.is_higher_metric_better(),
    )
    logger.info(
        "Strategy: %s  clients: %d  rounds: %d",
        getattr(cfg.strategy, "name", "random"),
        num_clients,
        cfg.num_server_rounds,
    )

    # Run federation
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=cfg.num_server_rounds,
        evaluate_fn=_get_evaluate_fn(
            cfg, task_adapter, val_set, data_collator, save_path
        ),
    )

    logger.info("Federation complete. Finalizing W&B run...")
    wandb_run.finish()
    logger.info("Done")


# -- Evaluation ----------------------------------------------------------------


def _get_validation_set(cfg, adapter: TaskAdapter) -> tuple[Dataset, Any]:
    """Load validation set using the task adapter."""
    dataset_name = cfg.dataset.name
    dataset_config = cfg.datasets[dataset_name]

    raw_val = load_data_centralized(dataset_config)

    encoding_fn = adapter.get_encoding_fn(cfg.model.name, dataset_config)
    data_collator = adapter.get_data_collator(model_cfg=cfg.model)
    val_set = raw_val.map(encoding_fn, batched=True)
    return val_set, data_collator


def _get_evaluate_fn(
    cfg: DictConfig,
    task_adapter: TaskAdapter,
    validation_set: Dataset,
    data_collator: Any,
    save_path: str,
):
    """Return evaluation closure for centralized evaluation."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        label = "pre-training" if server_round == 0 else f"round {server_round}"
        logger.info("[Eval %s] Running centralized evaluation...", label)

        import gc

        gc.collect()
        torch.cuda.empty_cache()

        model = task_adapter.get_model(cfg.model)
        set_peft_model_state_dict(model, arrays.to_torch_state_dict())

        # Periodically save model
        if server_round != 0 and (
            server_round == cfg.num_server_rounds
            or server_round % cfg.train.save_every_round == 0
        ):
            ckpt_path = f"{save_path}/peft_{server_round}"
            model.save_pretrained(ckpt_path)
            logger.info("[Eval %s] Checkpoint saved → %s", label, ckpt_path)

        # Optionally cap eval number
        eval_set = validation_set
        max_samples = cfg.eval.get("max_samples", -1)
        if max_samples > 0:
            limit = min(len(validation_set), max_samples)
            eval_set = validation_set.select(range(limit))
            logger.info("[Eval %s] Capped evaluation to %d samples", label, limit)

        # Evaluate
        trainer_args = TrainingArguments(
            output_dir=f"{save_path}/eval",
            per_device_eval_batch_size=cfg.eval.batch_size,
            report_to="none",  # strategy logs eval metrics on its own W&B step
        )
        trainer = Trainer(
            model=model,
            args=trainer_args,
            eval_dataset=eval_set,
            compute_metrics=task_adapter.compute_metrics,
            data_collator=data_collator,
        )
        metrics = trainer.evaluate()

        model.to("cpu")
        del trainer
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"[Eval {label}] Result: {metrics}")

        return MetricRecord(metrics)

    return evaluate


def _initialize_wandb(cfg):
    """Initialize W&B with experiment metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    dataset_name = cfg.dataset.name
    strategy_name = cfg.strategy.name
    seed = cfg.seed
    run_name_suffix = cfg.wandb.run_name
    run_name = f"{strategy_name}-{dataset_name}-{seed}"
    if run_name_suffix:
        run_name += f"-{run_name_suffix}"
    run_name += f"-{timestamp}"

    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        reinit=True,
    )


def _build_client_states(num_clients: int) -> dict[int, ClientState]:
    """Create an empty ClientState registry.

    All profiles start as None — each strategy populates them
    at runtime through the appropriate discovery mechanism (resource requests
    for FedCS, participation feedback for Oort, profiling rounds for TiFL).
    """
    return {cid: ClientState(client_id=cid) for cid in range(num_clients)}
