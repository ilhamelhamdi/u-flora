import math

import torch
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
    """HuggingFace Trainer with FedProx proximal regularization.

    Adds mu/2 * ||w - w^t||^2 to the standard task loss each step,
    where w^t are the frozen global parameters received from the server.
    """

    def __init__(
        self,
        *args,
        global_params: list[torch.Tensor],
        mu: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mu = mu
        # CPU copies of the initial (requires_grad) parameters; moved to the
        # training device lazily inside compute_loss via .to(device).
        self._global_params = global_params

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
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