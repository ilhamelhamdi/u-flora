import math

import torch
import torch.nn.functional as F
from transformers import Trainer


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Cosine annealing learning rate schedule.

    Decays learning rate from ``lrate_max`` to ``lrate_min`` over
    ``total_round`` rounds following a cosine curve.
    """
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def warmup_then_cosine(
    current_round: int,
    total_rounds: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
    warmup_rounds: int = 0,
) -> float:
    """Linear warmup followed by cosine annealing.

    Rounds 1..warmup_rounds ramp linearly from lrate_min to lrate_max.
    Remaining rounds decay via cosine annealing back to lrate_min.
    Falls back to plain cosine annealing when warmup_rounds == 0.
    """
    if warmup_rounds > 0 and current_round <= warmup_rounds:
        return lrate_min + (lrate_max - lrate_min) * (current_round / warmup_rounds)
    adjusted_round = current_round - warmup_rounds
    adjusted_total = max(total_rounds - warmup_rounds, 1)
    return cosine_annealing(adjusted_round, adjusted_total, lrate_max, lrate_min)


class FedProxTrainer(Trainer):
    """HuggingFace Trainer with FedProx proximal regularization and RMS loss tracking.

    Adds mu/2 * ||w - w^t||^2 to the standard task loss each step,
    where w^t are the frozen global parameters received from the server.

    When mu=0 this degrades to a standard Trainer.
    The per-sample RMS loss is accumulated across all training steps and exposed via the `loss_rms` property after trainer.train() completes.

    RMS loss formula (Oort Eq. 1):
        loss_rms = sqrt( (1/N) * sum(loss(k)^2 for k in B) )
    where N is the total number of valid (non-padding) tokens/samples seen.

    Args:
        global_params: Frozen copy of global model parameters for proximal term.
            Pass an empty list (or list of zeros) when mu=0 — the term vanishes.
        mu: FedProx proximal coefficient. 0.0 disables regularization entirely.
    """

    def __init__(
        self,
        *args,
        global_params: list[torch.Tensor],
        mu: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mu = mu
        self._global_params = global_params
        self._loss_sq_sum: float = 0.0
        self._loss_n: int = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # Single forward pass — outputs contain both logits and the model's
        # own mean loss (used for optimisation and Trainer's training_loss).
        outputs = model(**inputs)
        loss = outputs.loss  # model's mean loss, used for backprop

        # --- Per-sample RMS accumulation -----------------------------------
        # We re-use outputs.logits (already computed) to get per-sample losses
        # without a second forward pass. Only possible when labels are present
        # and the model produces logits (SEQ_CLS, causal LM, etc.).
        if (
            labels is not None
            and hasattr(outputs, "logits")
            and outputs.logits is not None
        ):
            logits = outputs.logits
            # Flatten to (batch*seq, vocab/num_labels) and (batch*seq,)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            # ignore_index=-100 is the HuggingFace convention for padding
            per_sample_loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                reduction="none",
            )
            # Only count positions that weren't masked out
            valid_mask = flat_labels != -100
            if valid_mask.any():
                self._loss_sq_sum += (
                    per_sample_loss[valid_mask].detach().pow(2).sum().item()
                )
                self._loss_n += valid_mask.sum().item()

        # --- FedProx proximal term -----------------------------------------
        if self.mu > 0.0 and self._global_params:
            device = loss.device
            prox = sum(
                ((p - g.to(device)) ** 2).sum()
                for p, g in zip(
                    (p for p in model.parameters() if p.requires_grad),
                    self._global_params,
                )
            )
            loss = loss + (self.mu / 2.0) * prox

        return (loss, outputs) if return_outputs else loss

    @property
    def loss_rms(self) -> float:
        """Per-sample RMS loss accumulated over all training steps.

        Returns 0.0 if no samples were seen (e.g., labels were not provided).
        """
        if self._loss_n == 0:
            return 0.0
        return math.sqrt(self._loss_sq_sum / self._loss_n)
