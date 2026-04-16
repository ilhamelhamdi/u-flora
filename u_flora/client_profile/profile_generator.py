"""Generate unified DeviceProfiles from heterogeneous trace data.

Combines two trace sources:
  1. Network traces (MobiPerf): bandwidth, latency, jitter
  2. Compute traces (Oort/FedScale): per-sample computation latency

This approach preserves the network characteristics as a single profile
unit, rather than destructing them. A portion of the combination will
be correlated (faster network mapped to faster compute), while the rest
are assigned randomly to increase variance and simulate real-world mismatches.

Usage:
    profiles, fedscale_t_min_ms = generate_device_profiles(
        num_profiles=100,
        network_trace_path="mobiperf.json",
        compute_trace_path="oort_client_device_capacity.json",
    )
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .typing import DeviceProfile


def _load_network_profiles(trace_path: str | Path | None) -> List[Dict[str, Any]]:
    """Load network profiles directly as objects and sort by capability."""
    with open(trace_path) as f:
        data = json.load(f)

    profiles = []
    for _, record in data.items():
        profiles.append(
            {
                "download_kbps": float(record.get("download_kbps", 0)),
                "upload_kbps": float(record.get("upload_kbps", 0)),
                "latency_ms": float(record.get("latency_ms", 0)),
                "jitter_ms": float(record.get("jitter_ms", 0)),
                "network_type": record.get("network_type", "unknown"),
            }
        )

    # Sort primarily by download bandwidth to estimate capability level (worst to best)
    profiles.sort(key=lambda p: p["download_kbps"])
    return profiles


def _load_compute_profiles(trace_path: str | Path | None) -> List[float]:
    """Extract computation latency values from Oort's client_device_capacity.

    Returns:
        (sorted_latencies, global_t_min_ms) where:
          - sorted_latencies: sampled list, ascending.
          - global_t_min_ms: minimum over the FULL trace.
    """
    with open(trace_path) as f:
        data = json.load(f)

    # Build full latency list to anchor the global minimum
    items = list(data.values())
    latencies: List[float] = []
    for caps in items:
        comp = caps.get("computation")
        if comp is not None and float(comp) > 0:
            latencies.append(float(comp))

    latencies.sort()
    return latencies


def generate_device_profiles(
    num_profiles: int,
    network_trace_path: str | Path | None = None,
    compute_trace_path: str | Path | None = None,
    seed: int = 42,
    correlated_ratio: float = 0.7,
) -> Tuple[List[DeviceProfile], float]:
    """Generate N device profiles by sampling from trace distributions.

    Args:
        num_profiles: Number of device profiles to generate.
        network_trace_path: Path to structured JSON trace for network.
        compute_trace_path: Path to Oort client_device_capacity JSON.
        seed: Random seed for reproducibility.
        correlated_ratio: Between 0.0 and 1.0. Determine proportion of dataset
            to correlate networking capability with compute capability.

    Returns:
        (profiles, fedscale_t_min_ms) where:
          - profiles: list of DeviceProfile instances.
          - fedscale_t_min_ms: global minimum computation latency from the full
            FedScale trace (pre-sampling). Used as the normalization anchor in
            runtime compute delay calibration. See _inject_compute_delay().
    """
    rng = random.Random(seed)

    net_profiles = _load_network_profiles(network_trace_path)
    compute_latencies = _load_compute_profiles(compute_trace_path)

    profiles: List[DeviceProfile] = []

    for i in range(num_profiles):
        is_correlated = rng.random() < correlated_ratio

        if is_correlated:
            # Single capability quantile: q in [0, 1]
            q = rng.random()

            # High q = better network (higher bandwidth index)
            net_idx = int(q * (len(net_profiles) - 1))
            net_idx = max(0, min(net_idx, len(net_profiles) - 1))
            net_prof = net_profiles[net_idx]

            # High q = better compute (lower latency index)
            # Since compute_latencies sorted ascending, lower index = better.
            # So if q=1 (best), idx=0. If q=0 (worst), idx=len-1
            comp_idx = int((1.0 - q) * (len(compute_latencies) - 1))
            comp_idx = max(0, min(comp_idx, len(compute_latencies) - 1))
            comp_val = compute_latencies[comp_idx]
        else:
            net_prof = rng.choice(net_profiles)
            comp_val = rng.choice(compute_latencies)

        # Add optional variance around computation latency specifically matching the paper
        noise_factor = rng.uniform(0.8, 1.2)
        final_latency = max(0.01, comp_val * noise_factor)

        profiles.append(
            DeviceProfile(
                client_id=i,
                computation_latency_ms=final_latency,
                download_kbps=net_prof["download_kbps"],
                upload_kbps=net_prof["upload_kbps"],
                latency_ms=net_prof["latency_ms"],
                jitter_ms=net_prof["jitter_ms"],
                network_type=net_prof["network_type"],
                device_name=f"device_{i}",
            )
        )

    fedscale_t_min_ms = min(compute_latencies)

    return profiles, fedscale_t_min_ms
