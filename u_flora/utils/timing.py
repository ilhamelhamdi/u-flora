"""Discrete-event duration estimation for FedScale-shaped device profiles.

A device profile in the FedScale trace shape has just two fields:

- ``computation`` — per-sample compute cost in ms (taken from the AI-Benchmark inference latency, used directly per the FedScale and Oort paper convention).
- ``communication`` — symmetric bandwidth in kbps.

We use these values directly. Strategy comparisons are scale-invariant and absolute durations stay in FedScale's published units.
"""

from __future__ import annotations


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
