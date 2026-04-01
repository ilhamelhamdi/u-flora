"""Generate unified DeviceProfiles from heterogeneous trace data.

Combines two trace sources:
  1. Network traces (MobiPerf): bandwidth, latency, jitter
  2. Compute traces (Oort/FedScale): per-sample computation latency

This approach preserves the network characteristics as a single profile
unit, rather than destructing them. A portion of the combination will
be correlated (faster network mapped to faster compute), while the rest
are assigned randomly to increase variance and simulate real-world mismatches.

Usage:
    profiles = generate_device_profiles(
        num_profiles=100,
        network_trace_path="mobiperf.json",
        compute_trace_path="oort_client_device_capacity.json",
    )
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from .base import DeviceProfile


def _load_network_profiles(trace_path: str | Path | None) -> List[Dict[str, Any]]:
    """Load network profiles directly as objects and sort by capability."""
    if not trace_path:
        # Fallback pseudo-profiles
        return [
            {
                "download_kbps": bw,
                "upload_kbps": bw * 0.5,
                "latency_ms": 10000 / bw,
                "jitter_ms":  2000 / bw,
                "network_type": "WIFI" if bw > 5000 else "LTE"
            }
            for bw in [100, 200, 500, 1000, 2000, 5000, 10000, 50000]
        ]

    with open(trace_path) as f:
        data = json.load(f)

    profiles = []
    for client_id, record in data.items():
        profiles.append({
            "download_kbps": float(record.get("download_kbps", 0)),
            "upload_kbps": float(record.get("upload_kbps", 0)),
            "latency_ms": float(record.get("latency_ms", 0)),
            "jitter_ms": float(record.get("jitter_ms", 0)),
            "network_type": record.get("network_type", "unknown")
        })

    # Sort primarily by download bandwidth to estimate capability level (worst to best)
    profiles.sort(key=lambda p: p["download_kbps"])
    return profiles


def _load_compute_profiles(trace_path: str | Path | None, sample_size: int | None = None, rng: random.Random | None = None) -> List[float]:
    """Extract computation latency values from Oort's client_device_capacity.

    Returns:
        Sorted list of computation_latency_ms values (fastest/best to slowest/worst).
                 i.e., smallest latency to largest latency.
    """
    if not trace_path:
        # Fallback distribution
        return [10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 500, 800]

    with open(trace_path) as f:
        data = json.load(f)

    # Convert to list to sample efficiently
    items = list(data.values())
    if sample_size and sample_size < len(items):
        if not rng:
            rng = random.Random(42)
        items = rng.sample(items, sample_size)

    latencies = []
    for caps in items:
        comp = caps.get("computation")
        if comp is not None and float(comp) > 0:
            latencies.append(float(comp))

    # Sort ascending: lowest latency (best compute) -> highest latency (worst compute)
    return sorted(latencies)


def generate_device_profiles(
    num_profiles: int,
    network_trace_path: str | Path | None = None,
    compute_trace_path: str | Path | None = None,
    seed: int = 42,
    correlated_ratio: float = 0.7,
) -> list[DeviceProfile]:
    """Generate N device profiles by sampling from trace distributions.

    Args:
        num_profiles: Number of device profiles to generate.
        network_trace_path: Path to structured JSON trace for network.
        compute_trace_path: Path to Oort client_device_capacity JSON.
        seed: Random seed for reproducibility.
        correlated_ratio: Between 0.0 and 1.0. Determine proportion of dataset
            to correlate networking capability with compute capability.

    Returns:
        List of DeviceProfile instances.
    """
    rng = random.Random(seed)

    net_profiles = _load_network_profiles(network_trace_path)
    compute_latencies = _load_compute_profiles(
        compute_trace_path, sample_size=num_profiles, rng=rng)

    profiles: list[DeviceProfile] = []

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

    return profiles
