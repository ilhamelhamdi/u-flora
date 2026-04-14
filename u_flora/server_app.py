"""Flower ServerApp — task-agnostic with pluggable client selection."""

import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from omegaconf import DictConfig, OmegaConf
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from .tasks import TaskAdapter
from .tasks.registry import get_task_adapter
from .dataset import load_data_centralized
from .utils import replace_keys
from .strategy import SelectionStrategy
from .selection.base import ClientState, DeviceProfile, ClientSelector
from .selection import RandomSelector

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

    # Initialize W&B
    wandb_run = _initialize_wandb(cfg)

    # Resolve task adapter
    task_name = cfg.task_name
    task_adapter = get_task_adapter(task_name)

    # Get initial model weights
    init_model = task_adapter.get_model(cfg.model)
    arrays = ArrayRecord(get_peft_model_state_dict(init_model))

    # Prepare validation set
    val_set, data_collator = _get_validation_set(cfg, task_adapter)

    # Build client selector and state registry
    num_clients = cfg.num_clients
    selector = _build_selector(cfg)
    client_states = _build_client_states(num_clients)

    logger.info(
        "Strategy: %s, Clients: %d, Rounds: %d",
        selector.name,
        num_clients,
        cfg.num_server_rounds,
    )

    # Define strategy
    strategy = SelectionStrategy(
        selector=selector,
        client_states=client_states,
        fraction_train=cfg.fraction_train,
        fraction_evaluate=0.0,
        use_wandb=True,
    )

    # Run federation
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        num_rounds=cfg.num_server_rounds,
        evaluate_fn=_get_evaluate_fn(
            cfg, task_adapter, val_set, data_collator, save_path
        ),
    )

    wandb_run.finish()


# -- Evaluation ----------------------------------------------------------------


def _get_validation_set(cfg, adapter):
    """Load validation set using the task adapter."""
    dataset_name = cfg.dataset_name
    dataset_config = cfg.datasets[dataset_name]

    raw_val = load_data_centralized(dataset_config)

    encoding_fn = adapter.get_encoding_fn(cfg.model.name, dataset_config)
    data_collator = adapter.get_data_collator(cfg.model.name)
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
        model = task_adapter.get_model(cfg.model)
        set_peft_model_state_dict(model, arrays.to_torch_state_dict())

        # Periodically save model
        if server_round != 0 and (
            server_round == cfg.num_server_rounds
            or server_round % cfg.train.save_every_round == 0
        ):
            model.save_pretrained(f"{save_path}/peft_{server_round}")

        # Evaluate
        trainer_args = TrainingArguments(
            output_dir=f"{save_path}/eval",
            per_device_eval_batch_size=cfg.eval.batch_size,
        )
        trainer = Trainer(
            model=model,
            args=trainer_args,
            eval_dataset=validation_set,
            compute_metrics=task_adapter.compute_metrics,
            data_collator=data_collator,
        )
        metrics = trainer.evaluate()

        # Log to W&B
        # TODO: can use task adapter to avoid hardcoding metric keys here
        # TODO: move W&B logging to strategy(?)
        log_dict = {"server_round": server_round, "eval_loss": metrics["eval_loss"]}
        if "eval_accuracy" in metrics:
            log_dict["eval_accuracy"] = metrics["eval_accuracy"]
        if "eval_perplexity" in metrics:
            log_dict["eval_perplexity"] = metrics["eval_perplexity"]
        wandb.log(log_dict)

        # Save to CSV
        res = {"round": [server_round]}
        res.update({k: [v] for k, v in metrics.items() if isinstance(v, (int, float))})
        df = pd.DataFrame(res)
        csv_path = f"{save_path}/results.csv"
        df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))

        return MetricRecord(metrics)

    return evaluate


def _initialize_wandb(cfg):
    """Initialize W&B with experiment metadata."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_name = getattr(cfg, "task", cfg.dataset)
    strategy_name = getattr(cfg.strategy, "name", "unknown")
    run_name_suffix = getattr(cfg.wandb, "run_name", "")
    run_name = f"{strategy_name}-{task_name}-{timestamp}"
    if run_name_suffix:
        run_name += f"-{run_name_suffix}"

    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
    )


# -- Selector Factory ----------------------------------------------------------


def _build_selector(cfg: DictConfig) -> ClientSelector:
    """Instantiate the configured ClientSelector.

    Reads from ``cfg.strategy`` which maps to one of the configs (name: `random`|`fedcs`|`oort`).
    Note: the selector is only used for training rounds. Evaluation is centralized and runs on the server, so it doesn't use the selector.
    """
    strategy_cfg = cfg.strategy
    name = str(getattr(strategy_cfg, "name", "random")).lower()
    num_to_select = int(getattr(strategy_cfg, "num_to_select", 10))
    seed = int(getattr(strategy_cfg, "seed", 42))

    if name == "random":
        return RandomSelector(num_to_select=num_to_select, seed=seed)
    else:
        logger.warning("Unknown strategy '%s', falling back to Random", name)
        return RandomSelector(num_to_select=num_to_select, seed=seed)


def _build_client_states(
    num_clients: int,
    profile_dir: str = "device_profiles",
) -> dict[int, ClientState]:
    """Build ClientState registry from generated device profiles.

    Reads from ``device_profiles/all_profiles.json`` if available,
    otherwise creates default profiles.
    """
    states: dict[int, ClientState] = {}
    profiles_path = os.path.join(profile_dir, "all_profiles.json")

    default_profile = DeviceProfile(
        client_id=-1,
        computation_latency_ms=50.0,
        download_kbps=5000.0,
        upload_kbps=3000.0,
        latency_ms=50.0,
        jitter_ms=5.0,
    )

    if os.path.exists(profiles_path):
        with open(profiles_path) as f:
            profile_dicts = json.load(f)
        logger.info(
            "Loaded %d device profiles from %s", len(profile_dicts), profiles_path
        )

        for pd_dict in profile_dicts[:num_clients]:
            cid = pd_dict["client_id"]
            if cid is None:
                logger.warning("Profile missing client_id, skipping entry: %s", pd_dict)
                continue

            if any(
                field not in pd_dict
                for field in [
                    "computation_latency_ms",
                    "download_kbps",
                    "upload_kbps",
                    "latency_ms",
                ]
            ):
                logger.warning(
                    "Profile for client_id %s is missing some fields, using defaults where needed",
                    cid,
                )

            profile = DeviceProfile(
                client_id=cid,
                computation_latency_ms=pd_dict.get(
                    "computation_latency_ms", default_profile.computation_latency_ms
                ),
                download_kbps=pd_dict.get(
                    "download_kbps", default_profile.download_kbps
                ),
                upload_kbps=pd_dict.get("upload_kbps", default_profile.upload_kbps),
                latency_ms=pd_dict.get("latency_ms", default_profile.latency_ms),
                jitter_ms=pd_dict.get("jitter_ms", default_profile.jitter_ms),
                network_type=pd_dict.get("network_type", default_profile.network_type),
                device_name=pd_dict.get("device_name", f"device_{cid}"),
            )
            states[cid] = ClientState(client_id=cid, profile=profile)
    else:
        logger.info(
            "No device profiles found at %s, using default profiles for %d clients",
            profiles_path,
            num_clients,
        )
        for cid in range(num_clients):
            profile = DeviceProfile(
                client_id=cid,
                computation_latency_ms=50.0,
                download_kbps=5000.0,
                upload_kbps=3000.0,
                latency_ms=50.0,
            )
            states[cid] = ClientState(client_id=cid, profile=profile)

    return states
