# ============================================================
#  notebook_2_h2_utility_signal.py
#
#  Purpose:
#    Preliminary evidence for H2:
#    "Fine-tuning from a low-loss pretrained initialization
#     compresses the dynamic range of Oort's statistical
#     utility signal."
#
#  Method:
#    Partition training data into N_CLIENTS virtual clients
#    (equal-sized shards). After each epoch, compute the mean
#    cross-entropy loss for each virtual client. Report the
#    Coefficient of Variation (CV = std/mean) across clients.
#
#    High CV = clients differ widely in loss = Oort can
#    discriminate useful clients.
#    Low CV  = clients look similar = Oort degenerates toward
#    random selection.
#
#    Run this for two conditions on the same dataset:
#      (A) LoRA from PRETRAINED init  — your actual FL setup
#      (B) Full fine-tuning from RANDOM init  — training-from-
#          scratch baseline (represents standard FL without
#          a pretrained foundation model)
#
#    Note on the baseline choice:
#    Condition B uses full parameter training from random init
#    (not LoRA-from-random-init) because that is the realistic
#    "training from scratch" scenario described in the Oort /
#    TiFL / FedCS papers. LoRA from random init is an artificial
#    setup that does not correspond to any real FL deployment.
#
#  H2 is supported if:
#    - Condition A (pretrained LoRA) shows CV collapsing to
#      near zero within the first 1-3 epochs.
#    - Condition B (full-model random init) shows CV staying
#      high for many more epochs.
#
#  Note: this is a PRELIMINARY / PROXY check only.
#  The definitive H2 evidence comes from the actual FL
#  experiments where per-client Oort utility scores are logged
#  to W&B across real federated rounds (see evaluation metrics
#  in §3.6 of the thesis).
# ============================================================


# ── Cell 0 : Install ─────────────────────────────────────────────────────────
# %%
# !pip install -q transformers peft datasets evaluate accelerate


# ── Cell 1 : Imports ─────────────────────────────────────────────────────────
# %%
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import evaluate as hf_evaluate

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Number of virtual clients used to approximate per-client loss distribution
N_CLIENTS = 100


# ── Cell 2 : Config (same as Notebook 1) ─────────────────────────────────────
# %%
@dataclass
class LoRAConfig:
    r: int               = 8
    lora_alpha: int      = 16
    lora_dropout: float  = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["Wqkv", "attn.Wo"]
    )
    bias: str = "none"


@dataclass
class TaskConfig:
    name: str
    hf_path: str
    hf_subset: str | None
    text_fields: list[str]
    label_field: str
    num_labels: int
    train_split: str     = "train"
    val_split: str       = "validation"
    model_name: str      = "answerdotai/ModernBERT-base"
    lora: LoRAConfig     = field(default_factory=LoRAConfig)
    learning_rate: float = 5e-4
    batch_size: int      = 32
    num_epochs: int      = 10


# ── Cell 3 : Choose dataset ───────────────────────────────────────────────────
# %%
# Use whichever task you selected as the primary FL benchmark from Notebook 1.
# Default: BoolQ. Change to "qnli" or "mnli" if that was your primary choice.

CHOSEN_TASK = TaskConfig(
    name        = "BoolQ (SuperGLUE)",
    hf_path     = "super_glue",
    hf_subset   = "boolq",
    text_fields = ["passage", "question"],
    label_field = "label",
    num_labels  = 2,
    num_epochs  = 10,
)

# Alternative — uncomment and swap if needed:
# CHOSEN_TASK = TaskConfig(
#     name="MNLI (GLUE)", hf_path="glue", hf_subset="mnli",
#     text_fields=["premise", "hypothesis"], label_field="label",
#     num_labels=3, val_split="validation_matched",
# )


# ── Cell 4 : Data loading ────────────────────────────────────────────────────
# %%
def tokenize_dataset(cfg: TaskConfig, raw_ds):
    """Tokenise a HuggingFace dataset using cfg.text_fields."""
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    def _encode(examples):
        parts = [examples[f] for f in cfg.text_fields]
        return tok(*parts, truncation=True, max_length=512)

    if cfg.label_field != "labels":
        raw_ds = raw_ds.rename_column(cfg.label_field, "labels")

    drop = [c for c in raw_ds.column_names
            if c not in cfg.text_fields + ["labels"]]
    ds = raw_ds.map(_encode, batched=True, remove_columns=drop)
    ds.set_format("torch")
    return ds


def load_splits(cfg: TaskConfig):
    raw_train = load_dataset(cfg.hf_path, cfg.hf_subset, split=cfg.train_split)
    raw_val   = load_dataset(cfg.hf_path, cfg.hf_subset, split=cfg.val_split)
    print(f"  Train: {len(raw_train):,} | Val: {len(raw_val):,}")
    return tokenize_dataset(cfg, raw_train), tokenize_dataset(cfg, raw_val)


# ── Cell 5 : Per-client loss CV computation ───────────────────────────────────
# %%
def compute_client_loss_cv(
    model: torch.nn.Module,
    train_ds,
    collator,
    n_clients: int = N_CLIENTS,
    batch_size: int = 64,
) -> tuple[float, list[float]]:
    """
    Partition train_ds into n_clients equal shards.
    Compute mean cross-entropy loss for each shard.
    Return (CV, per_client_losses).

    CV = std(losses) / mean(losses)

    This directly approximates Oort's statistical utility signal variance
    across the client pool.
    """
    model.eval()
    model.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    n       = len(train_ds)
    shard   = n // n_clients
    losses  = []

    with torch.no_grad():
        for i in range(n_clients):
            start = i * shard
            end   = (start + shard) if i < n_clients - 1 else n
            subset = train_ds.select(range(start, end))

            batch_losses = []
            for j in range(0, len(subset), batch_size):
                batch = collator(
                    [subset[k] for k in range(j, min(j + batch_size, len(subset)))]
                )
                batch  = {k: v.to(DEVICE) for k, v in batch.items()
                          if isinstance(v, torch.Tensor)}
                logits = model(**batch).logits
                loss   = loss_fn(logits, batch["labels"]).item()
                batch_losses.append(loss)
            losses.append(float(np.mean(batch_losses)))

    arr = np.array(losses)
    cv  = arr.std() / arr.mean() if arr.mean() > 0 else 0.0
    return float(cv), losses.copy()


# ── Cell 6 : CV-logging callback ─────────────────────────────────────────────
# %%
@dataclass
class CvPoint:
    epoch:          float
    cv:             float
    mean_loss:      float
    std_loss:       float
    min_loss:       float
    max_loss:       float
    val_accuracy:   float | None = None


class UtilityCvLogger(TrainerCallback):
    """
    After each epoch:
      1. Evaluates per-client loss CV on the training set.
      2. Records a CvPoint with full distribution statistics.
    """

    def __init__(self, train_ds, collator, n_clients: int = N_CLIENTS):
        self.train_ds  = train_ds
        self.collator  = collator
        self.n_clients = n_clients
        self.history: list[CvPoint] = []
        self._last_val_acc: float | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Capture latest val accuracy to attach to the CV point
        if metrics:
            self._last_val_acc = next(
                (v for k, v in metrics.items()
                 if "loss" not in k and isinstance(v, float)),
                None
            )

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        cv, per_client = compute_client_loss_cv(
            model, self.train_ds, self.collator, self.n_clients
        )
        arr = np.array(per_client)
        pt  = CvPoint(
            epoch        = state.epoch or 0.0,
            cv           = cv,
            mean_loss    = float(arr.mean()),
            std_loss     = float(arr.std()),
            min_loss     = float(arr.min()),
            max_loss     = float(arr.max()),
            val_accuracy = self._last_val_acc,
        )
        self.history.append(pt)
        print(f"    [CV] Epoch {pt.epoch:.0f} — "
              f"CV={cv:.4f}  mean_loss={pt.mean_loss:.4f}  "
              f"std={pt.std_loss:.4f}  "
              f"range=[{pt.min_loss:.4f}, {pt.max_loss:.4f}]")


# ── Cell 7 : Training runners ─────────────────────────────────────────────────
# %%
def make_lora_model(cfg: TaskConfig,
                    random_init: bool = False) -> torch.nn.Module:
    """
    random_init=False → pretrained ModernBERT + LoRA (your FL setup)
    random_init=True  → randomly initialised full model (train-from-scratch
                         baseline; all params trainable, no LoRA)
    """
    if random_init:
        # Randomly initialised — represents standard FL without a pretrained
        # foundation model (training from scratch scenario)
        config = AutoConfig.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_labels,
        )
        model = AutoModelForSequenceClassification.from_config(config)
        trainable = sum(p.numel() for p in model.parameters())
        total     = trainable
        print(f"  [random-init full model] "
              f"Trainable: {trainable:,} / {total:,} (100.00%)")
    else:
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_labels,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
        )
        peft_cfg = LoraConfig(
            task_type      = TaskType.SEQ_CLS,
            r              = cfg.lora.r,
            lora_alpha     = cfg.lora.lora_alpha,
            lora_dropout   = cfg.lora.lora_dropout,
            target_modules = cfg.lora.target_modules,
            bias           = cfg.lora.bias,
        )
        model = get_peft_model(base, peft_cfg)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"  [pretrained + LoRA] "
              f"Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)")
    return model


def run_condition(
    cfg: TaskConfig,
    train_ds,
    val_ds,
    collator,
    condition_label: str,
    random_init: bool,
) -> UtilityCvLogger:
    print(f"\n  ── Condition: {condition_label} ──")
    model  = make_lora_model(cfg, random_init=random_init)
    cv_log = UtilityCvLogger(train_ds, collator, n_clients=N_CLIENTS)

    metric_fn = hf_evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric_fn.compute(predictions=preds, references=labels)

    args = TrainingArguments(
        output_dir                  = f"/tmp/h2_{condition_label}",
        num_train_epochs            = cfg.num_epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        learning_rate               = cfg.learning_rate,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        eval_strategy               = "epoch",
        save_strategy               = "no",
        logging_strategy            = "epoch",
        report_to                   = "none",
        seed                        = SEED,
        fp16                        = (torch.cuda.is_available()
                                       and not random_init),
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = compute_metrics,
        callbacks       = [cv_log],
    )
    trainer.train()
    return cv_log


# ── Cell 8 : Main run ─────────────────────────────────────────────────────────
# %%
cfg         = CHOSEN_TASK
train_ds, val_ds = load_splits(cfg)
tok         = AutoTokenizer.from_pretrained(cfg.model_name)
collator    = DataCollatorWithPadding(tokenizer=tok)

print(f"\nTask: {cfg.name}")
print(f"Virtual clients for CV: {N_CLIENTS}")

# Condition A — pretrained LoRA (your actual FL setup)
log_pretrained = run_condition(
    cfg, train_ds, val_ds, collator,
    condition_label="pretrained_lora",
    random_init=False,
)

# Condition B — full model from random init (train-from-scratch baseline)
log_random = run_condition(
    cfg, train_ds, val_ds, collator,
    condition_label="random_init_full",
    random_init=True,
)


# ── Cell 9 : Plotting ─────────────────────────────────────────────────────────
# %%
def plot_h2(log_pt: UtilityCvLogger, log_rnd: UtilityCvLogger,
            task_name: str, figsize=(14, 10)):
    """
    Four-panel figure:
      Row 1 — CV over epochs + loss distribution box plots
      Row 2 — Mean loss over epochs + val accuracy over epochs
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax_cv, ax_box), (ax_loss, ax_acc) = axes

    col_pt  = "#1f77b4"  # blue  — pretrained LoRA
    col_rnd = "#d62728"  # red   — random init

    ep_pt  = [p.epoch for p in log_pt.history]
    ep_rnd = [p.epoch for p in log_rnd.history]

    # ── CV vs epoch ───────────────────────────────────────────────────────────
    ax_cv.plot(ep_pt,  [p.cv for p in log_pt.history],
               "o-", color=col_pt,  ms=6, label="Pretrained + LoRA")
    ax_cv.plot(ep_rnd, [p.cv for p in log_rnd.history],
               "s-", color=col_rnd, ms=6, label="Random init (full model)")
    ax_cv.axhline(0.05, color="gray", ls="--", lw=0.8,
                  label="CV=0.05 (low discrimination)")
    ax_cv.set_xlabel("Epoch"); ax_cv.set_ylabel("Loss CV (std / mean)")
    ax_cv.set_title("H2 — Oort utility signal CV over training")
    ax_cv.legend(); ax_cv.grid(True, alpha=0.3); ax_cv.set_ylim(bottom=0)

    # ── Loss range box plot at epoch 1 and final epoch ────────────────────────
    def get_client_losses_at_epoch(log, target_epoch):
        # Re-compute is not stored per-client; approximate with range bars
        pt = min(log.history, key=lambda p: abs(p.epoch - target_epoch))
        return pt.mean_loss, pt.std_loss, pt.min_loss, pt.max_loss

    for ep_target, offset, label, col in [
        (1.0, -0.2, "Ep 1", col_pt),
        (min(len(log_pt.history), 10), 0.2, f"Ep {len(log_pt.history)}", col_pt),
    ]:
        m, s, lo, hi = get_client_losses_at_epoch(log_pt, ep_target)
        ax_box.errorbar(offset, m, yerr=[[m-lo], [hi-m]],
                        fmt="o", color=col, capsize=5,
                        label=f"Pretrained LoRA {label}")
    for ep_target, offset, label, col in [
        (1.0, 0.6, "Ep 1", col_rnd),
        (min(len(log_rnd.history), 10), 1.0, f"Ep {len(log_rnd.history)}", col_rnd),
    ]:
        m, s, lo, hi = get_client_losses_at_epoch(log_rnd, ep_target)
        ax_box.errorbar(offset, m, yerr=[[m-lo], [hi-m]],
                        fmt="s", color=col, capsize=5,
                        label=f"Random init {label}")
    ax_box.set_ylabel("Per-client mean loss (range)")
    ax_box.set_title("Loss distribution: early vs late training")
    ax_box.set_xticks([]); ax_box.legend(fontsize=8); ax_box.grid(True, alpha=0.3)

    # ── Mean loss vs epoch ────────────────────────────────────────────────────
    ax_loss.plot(ep_pt,  [p.mean_loss for p in log_pt.history],
                 "o-", color=col_pt,  ms=5, label="Pretrained + LoRA")
    ax_loss.plot(ep_rnd, [p.mean_loss for p in log_rnd.history],
                 "s-", color=col_rnd, ms=5, label="Random init (full model)")
    ax_loss.fill_between(
        ep_pt,
        [p.mean_loss - p.std_loss for p in log_pt.history],
        [p.mean_loss + p.std_loss for p in log_pt.history],
        alpha=0.15, color=col_pt,
    )
    ax_loss.fill_between(
        ep_rnd,
        [p.mean_loss - p.std_loss for p in log_rnd.history],
        [p.mean_loss + p.std_loss for p in log_rnd.history],
        alpha=0.15, color=col_rnd,
    )
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Mean ± std loss across clients")
    ax_loss.set_title("Per-client loss: mean ± std (shaded = 1σ)")
    ax_loss.legend(); ax_loss.grid(True, alpha=0.3); ax_loss.set_ylim(bottom=0)

    # ── Val accuracy vs epoch ─────────────────────────────────────────────────
    acc_pt  = [p.val_accuracy for p in log_pt.history
               if p.val_accuracy is not None]
    acc_rnd = [p.val_accuracy for p in log_rnd.history
               if p.val_accuracy is not None]
    ax_acc.plot(ep_pt[:len(acc_pt)],   acc_pt,
                "o-", color=col_pt,  ms=5, label="Pretrained + LoRA")
    ax_acc.plot(ep_rnd[:len(acc_rnd)], acc_rnd,
                "s-", color=col_rnd, ms=5, label="Random init (full model)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax_acc.set_title("Validation accuracy (context for H1)")
    ax_acc.legend(); ax_acc.grid(True, alpha=0.3); ax_acc.set_ylim(bottom=0)

    fig.suptitle(
        f"H2 Preliminary Evidence — {task_name}\n"
        f"Virtual clients: {N_CLIENTS}",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig("nb2_h2_utility_signal.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_h2_summary(log_pt: UtilityCvLogger, log_rnd: UtilityCvLogger):
    print("\n── H2 summary ─────────────────────────────────────────────────")
    print(f"{'Epoch':<8} {'CV (pretrained)':>18} {'CV (random init)':>18} {'Ratio':>8}")
    print("─" * 56)
    for pt, rnd in zip(log_pt.history, log_rnd.history):
        ratio = rnd.cv / pt.cv if pt.cv > 0 else float("inf")
        print(f"{pt.epoch:<8.0f} {pt.cv:>18.4f} {rnd.cv:>18.4f} {ratio:>8.2f}x")
    print()

    ep1_pt  = log_pt.history[0].cv  if log_pt.history  else None
    ep1_rnd = log_rnd.history[0].cv if log_rnd.history else None
    if ep1_pt and ep1_rnd:
        ratio = ep1_rnd / ep1_pt
        print(f"  At epoch 1: random-init CV is {ratio:.1f}x higher than pretrained.")
        verdict = (
            "SUPPORTED (>2x)"     if ratio > 2.0 else
            "WEAKLY SUPPORTED"    if ratio > 1.3 else
            "NOT SUPPORTED"
        )
        print(f"  H2: {verdict}")

    pt_below = next(
        (p.epoch for p in log_pt.history  if p.cv < 0.05), None
    )
    rnd_below = next(
        (p.epoch for p in log_rnd.history if p.cv < 0.05), None
    )
    print(f"\n  Pretrained  crosses CV<0.05 at epoch: {pt_below}")
    print(f"  Random init crosses CV<0.05 at epoch: {rnd_below}")


plot_h2(log_pretrained, log_random, cfg.name)
print_h2_summary(log_pretrained, log_random)


# %% [markdown]
# ## Interpretation
#
# **What H2 predicts you should see:**
# - Pretrained + LoRA: CV starts low (near 0) at epoch 1 because all
#   clients start from the same pretrained weights in a low-loss region.
#   Clients look similar to Oort from the very beginning.
# - Random init (full model): CV starts high at epoch 1 because randomly
#   initialised weights produce very different loss values depending on
#   each client's local data distribution. CV gradually decreases as
#   training progresses and the model learns the task.
#
# **If H2 is NOT supported (both conditions show similar CV):**
# This could mean your virtual-client shards do not have enough
# label/feature diversity to create meaningfully different loss values.
# Consider using a more non-IID partition (e.g. sort by label then shard)
# to simulate a more realistic FL data distribution.
#
# **Relationship to the main FL experiment:**
# This notebook is a proxy using virtual clients. The definitive evidence
# for H2 comes from the Oort utility CV logged per round in the main FL
# experiments (already implemented in u_flora as strategy/utility_cv).