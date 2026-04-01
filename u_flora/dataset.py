from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate
import numpy as np
from omegaconf import DictConfig

FDS = None  # Cache FederatedDataset


def encoding_func(dataset_config, examples, tokenizer):
    text_fields = dataset_config["text_fields"].split(",")
    inputs = [examples[field] for field in text_fields]
    return tokenizer(*inputs, truncation=True)


def get_encoding_func_and_data_collator(model_name: str, dataset_config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def enc(examples): return encoding_func(dataset_config, examples, tokenizer)
    return enc, data_collator


def load_data(partition_id: int, num_partitions: int, dataset_config: DictConfig):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        # TODO: Later change to DirichletPartitioner or other partitioning strategy to simulate non-IID data distribution
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        val_partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_config.name,
            subset=dataset_config.subset,
            partitioners={"train": train_partitioner, "validation": val_partitioner},
        )
    client_train_set = FDS.load_partition(partition_id, "train")
    client_val_set = FDS.load_partition(partition_id, "validation")

    return client_train_set, client_val_set


accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    return {
        "accuracy": acc["accuracy"],
    }
