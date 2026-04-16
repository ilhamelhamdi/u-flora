"""
Measure per-sample training time for ModernBERT + LoRA on each GLUE task.

Run this script once on the DGX before experiments to populate
T_ACTUAL_MS_PER_SAMPLE in configs/compute_baseline.py.

Usage:
    python scripts/profile_compute_baseline.py --task sst2 --batch_size 32 --warmup_steps 5 --measure_steps 20
    python scripts/profile_compute_baseline.py --task qnli --batch_size 32
    python scripts/profile_compute_baseline.py --task qqp  --batch_size 32
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# ── Task definitions ──────────────────────────────────────────────────────────

TASK_CONFIG: dict[str, dict] = {
    "sst2": {
        "hf_path": ("glue", "sst2"),
        "text_fields": ["sentence"],
        "num_labels": 2,
    },
    "qnli": {
        "hf_path": ("glue", "qnli"),
        "text_fields": ["question", "sentence"],
        "num_labels": 2,
    },
    "qqp": {
        "hf_path": ("glue", "qqp"),
        "text_fields": ["question1", "question2"],
        "num_labels": 2,
    },
}

MODEL_NAME = "answerdotai/ModernBERT-base"

# LoRA config
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["Wqkv", "attn.Wo"],
    bias="none",
    task_type="SEQ_CLS",
)

# ── Profiling ─────────────────────────────────────────────────────────────────


def profile_task(
    task: str,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> None:
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_CONFIG)}")

    cfg = TASK_CONFIG[task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {task}  batch_size={batch_size}  warmup={warmup_steps}  measure={measure_steps}")

    # -- Load tokenizer and model ------------------------------------------
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=cfg["num_labels"]
    )
    model = get_peft_model(base_model, LORA_CONFIG)
    model.to(device)
    model.train()

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # -- Load a small slice of the dataset ---------------------------------
    print("Loading dataset...")
    hf_path = cfg["hf_path"]
    raw = load_dataset(*hf_path, split="train")

    # Only need enough samples to fill warmup + measure batches
    needed = (warmup_steps + measure_steps) * batch_size
    raw = raw.select(range(min(needed, len(raw))))

    text_fields = cfg["text_fields"]

    def tokenize(batch):
        if len(text_fields) == 1:
            return tokenizer(
                batch[text_fields[0]],
                truncation=True,
                max_length=128,
                padding=False,
            )
        return tokenizer(
            batch[text_fields[0]],
            batch[text_fields[1]],
            truncation=True,
            max_length=128,
            padding=False,
        )

    encoded = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(encoded, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # -- Warmup passes (discarded) -----------------------------------------
    print(f"Running {warmup_steps} warmup steps...")
    loader_iter = iter(loader)
    for _ in range(warmup_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # -- Timed measurement passes ------------------------------------------
    print(f"Running {measure_steps} timed steps...")
    step_times_ms: list[float] = []

    for step in range(measure_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        actual_batch_size = batch["input_ids"].shape[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        ms_per_sample = elapsed_ms / actual_batch_size
        step_times_ms.append(ms_per_sample)

    # -- Report results ----------------------------------------------------
    mean_ms = statistics.mean(step_times_ms)
    std_ms = statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0

    print()
    print("=" * 50)
    print(f"  Task:               {task}")
    print(f"  Batch size:         {batch_size}")
    print(f"  mean_ms_per_sample: {mean_ms:.4f}")
    print(f"  std_ms_per_sample:  {std_ms:.4f}")
    print(f"  Measure steps:      {measure_steps}")
    print("=" * 50)
    print()
    print("Paste into configs/compute_baseline.py:")
    print(f'    "{task}": {mean_ms:.4f},')


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile per-sample training time for ModernBERT+LoRA on a GLUE task."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_CONFIG),
        help="GLUE task to profile (sst2 | qnli | qqp)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for profiling")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps (discarded)")
    parser.add_argument("--measure_steps", type=int, default=20, help="Steps to time")
    args = parser.parse_args()

    profile_task(
        task=args.task,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )


if __name__ == "__main__":
    main()
