# ============================================================
#  notebook_3_compute_profiling.py
#
#  Purpose:
#    Measure t_actual — the wall-clock time for one training
#    step of ModernBERT + LoRA on THIS hardware, per task.
#
#    This value is required BEFORE running FL experiments.
#    It is the anchor for the compute delay formula:
#
#        t_compute_i = t_actual * (tau_i / tau_min)
#
#    Without measured t_actual values, the heterogeneity
#    simulation assigns delays based on arbitrary numbers.
#
#    Run this notebook ONCE on your DGX-A100 testbed (not Colab)
#    before starting any FL experiments.
#    Mirrors scripts/profile_compute_baseline.py in u_flora
#    but adds per-task breakdown and a summary table.
#
#  Output:
#    - ms_per_sample for each task
#    - Bar chart comparing compute time across tasks
#    - A Python dict ready to paste into
#      configs/compute_baseline.py in u_flora
# ============================================================


# ── Cell 0 : Install ─────────────────────────────────────────────────────────
# %%
# !pip install -q transformers peft datasets accelerate


# ── Cell 1 : Imports ─────────────────────────────────────────────────────────
# %%
from __future__ import annotations

import time
import statistics
import warnings
from dataclasses import dataclass, field

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

warnings.filterwarnings("ignore")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")


# ── Cell 2 : Config ───────────────────────────────────────────────────────────
# %%
MODEL_NAME  = "answerdotai/ModernBERT-base"
LORA_R      = 8
LORA_ALPHA  = 16
LORA_DROP   = 0.05
TARGET_MODS = ["Wqkv", "attn.Wo"]
BATCH_SIZE  = 32
WARMUP_STEPS   = 10   # discarded — let GPU warm up
MEASURE_STEPS  = 50   # steps to average over
MAX_SEQ_LEN    = 512


@dataclass
class ProbeConfig:
    """One entry per task to profile."""
    name: str
    hf_path: str
    hf_subset: str | None
    text_fields: list[str]
    label_field: str
    num_labels: int
    train_split: str = "train"


PROBE_TASKS: dict[str, ProbeConfig] = {
    "sst2": ProbeConfig(
        name="SST-2", hf_path="glue", hf_subset="sst2",
        text_fields=["sentence"], label_field="label", num_labels=2,
    ),
    "qnli": ProbeConfig(
        name="QNLI", hf_path="glue", hf_subset="qnli",
        text_fields=["question", "sentence"], label_field="label", num_labels=2,
    ),
    "mnli": ProbeConfig(
        name="MNLI", hf_path="glue", hf_subset="mnli",
        text_fields=["premise", "hypothesis"], label_field="label", num_labels=3,
    ),
    "boolq": ProbeConfig(
        name="BoolQ", hf_path="super_glue", hf_subset="boolq",
        text_fields=["passage", "question"], label_field="label", num_labels=2,
    ),
    "medmcqa": ProbeConfig(
        name="MedMCQA", hf_path="openlifescienceai/medmcqa", hf_subset=None,
        text_fields=["question", "opa", "opb", "opc", "opd"],
        label_field="cop", num_labels=4,
    ),
}


# ── Cell 3 : Model builder ───────────────────────────────────────────────────
# %%
def build_model(num_labels: int) -> torch.nn.Module:
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )
    peft_cfg = LoraConfig(
        task_type      = TaskType.SEQ_CLS,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROP,
        target_modules = TARGET_MODS,
        bias           = "none",
    )
    model = get_peft_model(base, peft_cfg)
    model.to(DEVICE)
    model.train()
    return model


# ── Cell 4 : Data loader ─────────────────────────────────────────────────────
# %%
def build_dataloader(cfg: ProbeConfig) -> tuple[DataLoader, int]:
    """Returns (dataloader, n_samples_in_one_batch)."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw = load_dataset(cfg.hf_path, cfg.hf_subset, split=cfg.train_split)

    # Keep only enough samples for WARMUP_STEPS + MEASURE_STEPS batches
    n_needed = (WARMUP_STEPS + MEASURE_STEPS + 5) * BATCH_SIZE
    if len(raw) > n_needed:
        raw = raw.shuffle(seed=SEED).select(range(n_needed))

    if cfg.label_field != "labels":
        raw = raw.rename_column(cfg.label_field, "labels")

    fields = cfg.text_fields

    def _encode(examples):
        parts = [examples[f] for f in fields]
        return tok(*parts, truncation=True, max_length=MAX_SEQ_LEN)

    drop = [c for c in raw.column_names
            if c not in fields + ["labels"]]
    ds   = raw.map(_encode, batched=True, remove_columns=drop)
    ds.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tok)
    loader   = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collator,
                          shuffle=True)
    return loader, BATCH_SIZE


# ── Cell 5 : Profiling function ───────────────────────────────────────────────
# %%
def profile_task(cfg: ProbeConfig) -> dict:
    """
    Run WARMUP_STEPS warm-up steps (discarded), then measure
    MEASURE_STEPS steps and return timing statistics.
    """
    print(f"\n  Profiling: {cfg.name}")
    model   = build_model(cfg.num_labels)
    loader, _  = build_dataloader(cfg)
    opt     = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    loader_iter = iter(loader)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"    Warming up ({WARMUP_STEPS} steps)…", end="", flush=True)
    for _ in range(WARMUP_STEPS):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch  = {k: v.to(DEVICE) for k, v in batch.items()
                  if isinstance(v, torch.Tensor)}
        logits = model(**batch).logits
        loss   = loss_fn(logits, batch["labels"])
        opt.zero_grad()
        loss.backward()
        opt.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    print(" done.")

    # ── Measurement ───────────────────────────────────────────────────────────
    step_times_ms: list[float] = []
    actual_batch_sizes: list[int] = []

    for i in range(MEASURE_STEPS):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        n_samples = batch["labels"].shape[0]
        batch = {k: v.to(DEVICE) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(**batch).logits
        loss   = loss_fn(logits, batch["labels"])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        step_times_ms.append(elapsed_ms)
        actual_batch_sizes.append(n_samples)

    # Per-sample time in ms
    per_sample_ms = [t / n for t, n in zip(step_times_ms, actual_batch_sizes)]

    result = {
        "task"                    : cfg.name,
        "device"                  : torch.cuda.get_device_name(0)
                                    if DEVICE == "cuda" else "cpu",
        "batch_size"              : BATCH_SIZE,
        "measure_steps"           : MEASURE_STEPS,
        "mean_step_ms"            : round(statistics.mean(step_times_ms), 2),
        "std_step_ms"             : round(statistics.stdev(step_times_ms), 2),
        "mean_ms_per_sample"      : round(statistics.mean(per_sample_ms), 4),
        "std_ms_per_sample"       : round(statistics.stdev(per_sample_ms), 4),
        "min_ms_per_sample"       : round(min(per_sample_ms), 4),
        "max_ms_per_sample"       : round(max(per_sample_ms), 4),
    }
    print(f"    mean={result['mean_ms_per_sample']:.4f} ms/sample  "
          f"(step avg={result['mean_step_ms']:.1f}ms ± "
          f"{result['std_step_ms']:.1f}ms)")

    # Clean up GPU memory between tasks
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result


# ── Cell 6 : Run all tasks ───────────────────────────────────────────────────
# %%
TASKS_TO_PROFILE = ["sst2", "boolq", "qnli", "mnli", "medmcqa"]

profile_results: dict[str, dict] = {}
for key in TASKS_TO_PROFILE:
    profile_results[key] = profile_task(PROBE_TASKS[key])


# ── Cell 7 : Report ───────────────────────────────────────────────────────────
# %%
def print_profile_table(results: dict):
    hdr = (f"{'Task':<12} {'ms/sample (mean)':>18} {'ms/sample (std)':>17} "
           f"{'step ms':>10} {'device'}")
    sep = "─" * 75
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results.values():
        print(
            f"{r['task']:<12} {r['mean_ms_per_sample']:>18.4f} "
            f"{r['std_ms_per_sample']:>17.4f} "
            f"{r['mean_step_ms']:>10.1f} "
            f"{r['device']}"
        )
    print(sep)


def plot_profile(results: dict):
    tasks  = [r["task"] for r in results.values()]
    means  = [r["mean_ms_per_sample"] for r in results.values()]
    stds   = [r["std_ms_per_sample"]  for r in results.values()]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(tasks, means, yerr=stds, capsize=5,
                  color="#4C72B0", edgecolor="white", width=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ms per training sample", fontsize=11)
    ax.set_title(
        f"ModernBERT + LoRA (r={LORA_R}) — compute time per task\n"
        f"Device: {list(results.values())[0]['device']}",
        fontsize=12
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("nb3_compute_profile.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_config_snippet(results: dict):
    """Print the dict to paste into configs/compute_baseline.py."""
    print("\n# Paste into configs/compute_baseline.py:")
    print("T_ACTUAL_MS_PER_SAMPLE: dict[str, float] = {")
    for key, r in results.items():
        print(f'    "{key}": {r["mean_ms_per_sample"]},')
    print("}")


print_profile_table(profile_results)
plot_profile(profile_results)
print_config_snippet(profile_results)


# %% [markdown]
# ## Notes
#
# **Why this matters:**
# The compute delay injected into each FL client is:
#
#     t_compute_i = t_actual * (tau_i / tau_min)
#
# where t_actual is the value measured here and tau_i / tau_min is the
# relative speed ratio from the FedScale trace. Without a measured
# t_actual, the simulation either over- or under-estimates how long
# each round takes in absolute wall-clock terms.
#
# **Why tasks differ:**
# BoolQ and MNLI have longer input sequences (passage + question) than
# SST-2 (single sentence), so each training step processes more tokens
# and takes longer per sample. MedMCQA concatenates 5 fields. These
# differences directly affect TTA comparisons across tasks.
#
# **Re-running:**
# If you change LORA_R or TARGET_MODS, re-run this notebook and update
# configs/compute_baseline.py before starting FL experiments.