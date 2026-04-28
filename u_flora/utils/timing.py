"""Discrete-event duration estimation for FedScale-shaped device profiles.

A device profile in the FedScale trace shape has just two fields:

- ``computation`` — per-sample compute cost in ms (taken from the AI-Benchmark inference latency, used directly per the FedScale and Oort paper convention).
- ``communication`` — symmetric bandwidth in kbps.


Execution-time jitter (`apply_duration_jitter`) is layered on top of the
deterministic estimate to model real-world noise — thermal throttling,
foreground app contention, network congestion. Strategies see the
estimator's deterministic prediction (as they would in real life from running
averages); the *realised* duration the client reports is jittered.
"""

from __future__ import annotations

import math
import random
from typing import Hashable


def estimate_round_duration_s(
    profile: dict,
    num_samples: int,
    local_epochs: int,
    model_size_kb: float,
) -> tuple[float, float, float]:
    """Estimate (compute_s, comm_s, total_s) for one round on one client.

    Args:
        profile: A FedScale-shaped profile dict with ``computation`` (ms/sample)
            and ``communication`` (kbps) keys.
        num_samples: Number of training samples on this client.
        local_epochs: Number of local training epochs.
        model_size_kb: Size of the model update payload in kilobytes.

    Returns:
        (compute_s, comm_s, total_s) — all in seconds.
    """
    raw_comp_ms = float(profile.get("computation", 0.0))
    comm_kbps = float(profile.get("communication", 0.0))

    if raw_comp_ms > 0:
        compute_s = raw_comp_ms * num_samples * local_epochs / 1000.0
    else:
        compute_s = 0.0

    if comm_kbps > 0 and model_size_kb > 0:
        # Symmetric: download the global model and upload the update.
        comm_s = 2.0 * model_size_kb / comm_kbps
    else:
        comm_s = 0.0

    return compute_s, comm_s, compute_s + comm_s


def apply_duration_jitter(
    compute_s: float,
    comm_s: float,
    *,
    compute_cv: float = 0.0,
    comm_cv: float = 0.0,
    compute_min_factor: float = 0.7,
    comm_min_factor: float = 0.5,
    comm_stall_prob: float = 0.0,
    comm_stall_factor_min: float = 5.0,
    comm_stall_factor_max: float = 10.0,
    seed: Hashable = 0,
) -> tuple[float, float, float, dict]:
    """Apply log-normal multiplicative jitter to compute and communication times.

    Compute jitter models thermal throttling and foreground contention.
    Comm jitter models cell handoffs, congestion; an extra ``comm_stall_prob``
    chance fires a heavier 5-10x tail event to capture network black-holing
    (this is the regime FedCS's deadline mechanism is meant to react to).

    The ``seed`` should encode (run_seed, round, partition_id) so the same
    experiment is reproducible while different (round, client) pairs see
    independent draws.

    Args:
        compute_s: Deterministic compute estimate in seconds.
        comm_s: Deterministic communication estimate in seconds.
        compute_cv: Coefficient of variation for compute jitter (0 disables).
        comm_cv: CV for comm jitter (0 disables).
        compute_min_factor: Lower clip for compute multiplier.
        comm_min_factor: Lower clip for comm multiplier.
        comm_stall_prob: Probability of a heavy comm-stall event per round.
        comm_stall_factor_min/max: Stall multiplier range (uniform).
        seed: Hashable seed for the per-call RNG.

    Returns:
        (compute_s, comm_s, total_s, metadata) — jittered seconds and a
        dict with the multipliers actually applied (for telemetry).
    """
    seed_val = seed if isinstance(seed, (int, str, bytes, bytearray, float)) else str(seed)
    rng = random.Random(seed_val)

    def _lognormal(cv: float, floor: float) -> float:
        if cv <= 0.0:
            return 1.0
        # Param such that the resulting lognormal has the requested CV.
        sigma = math.sqrt(math.log1p(cv * cv))
        mu = -0.5 * sigma * sigma  # ensures mean = 1
        m = rng.lognormvariate(mu, sigma)
        return max(floor, m)

    compute_mult = _lognormal(compute_cv, compute_min_factor)
    comm_mult = _lognormal(comm_cv, comm_min_factor)

    stall_mult = 1.0
    if comm_stall_prob > 0.0 and rng.random() < comm_stall_prob:
        stall_mult = rng.uniform(comm_stall_factor_min, comm_stall_factor_max)
        comm_mult *= stall_mult

    out_compute = compute_s * compute_mult
    out_comm = comm_s * comm_mult
    meta = {
        "jitter_compute_mult": compute_mult,
        "jitter_comm_mult": comm_mult,
        "jitter_comm_stall": stall_mult,
    }
    return out_compute, out_comm, out_compute + out_comm, meta
