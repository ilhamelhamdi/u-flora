"""flowertune-distilbert-lora: A Flower / FlowerTune app."""

import os
import warnings
import time

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, Trainer

from .dataset import (
    get_encoding_func_and_data_collator,
    load_data,
    compute_metrics,
)
from .utils import replace_keys
from .models import cosine_annealing, get_model

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Parse config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    num_rounds = cfg.num_server_rounds
    chosen_dataset = cfg.dataset
    dataset_config = cfg.datasets[chosen_dataset]
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    encoding_func, data_collator = get_encoding_func_and_data_collator(
        cfg.model.name, dataset_config)
    train_set, _ = load_data(partition_id, num_partitions, dataset_config)
    train_set = train_set.map(encoding_func, batched=True)

    # Load the model and initialize it with the received weights
    model = get_model(cfg.model)
    set_peft_model_state_dict(
        model, msg.content["arrays"].to_torch_state_dict())

    # Set learning rate for current round
    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
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
        data_collator=data_collator
    )

    # Do local training
    start_time = time.time()
    results = trainer.train()
    duration = time.time() - start_time

    # Construct and return reply Message
    model_record = ArrayRecord(get_peft_model_state_dict(model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(train_set),
        "duration": duration,
        "node_id": partition_id,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# We don't implement federated evalution, but centralized evaluation.
# @app.evaluate()
# def evaluate(msg: Message, context: Context):
#     """Evaluate the model on local data."""
#     # Parse config
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

#     # Let's get the client partition
#     encoding_func, data_collator = get_encoding_func_and_data_collator(
#         cfg.model.name)
#     _, val_set = load_data(
#         partition_id, num_partitions, cfg.dataset.name, cfg.dataset.subset)
#     val_set = val_set.map(encoding_func, batched=True)

#     # Load the model and initialize it with the received weights
#     model = get_model(cfg.model)
#     set_peft_model_state_dict(
#         model, msg.content["arrays"].to_torch_state_dict())

#     trainer = Trainer(
#         model=model,
#         eval_dataset=val_set,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )

#     # Do local evaluation
#     metrics = trainer.evaluate()

#     # Construct and return reply Message
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"metrics": metric_record})
#     return Message(content=content, reply_to=msg)
