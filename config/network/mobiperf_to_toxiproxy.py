"""
MobiPerf Trace → ToxiProxy Configuration Generator for Federated Learning

This script parses MobiPerf JSON trace data and generates per-client
ToxiProxy configurations that simulate realistic heterogeneous network
conditions in a cross-device FL setting.

Usage:
    python mobiperf_to_toxiproxy.py \
        --trace path/to/mobiperf.json \
        --num_clients 20 \
        --output toxiproxy_configs.json \
        --mode profile  # or 'sample'

Modes:
    - profile: Creates N client profiles by sampling from the empirical
               distribution (bandwidth, latency, jitter). Each client gets
               a fixed profile for the duration of training.
    - sample:  Creates per-round configurations where each client's network
               conditions can vary across rounds (models temporal variability).

ToxiProxy mapping:
    MobiPerf measurement     →  ToxiProxy toxic
    ─────────────────────────────────────────────
    tcp_speed_results (kbps) →  bandwidth (rate limit)
    mean_rtt_ms              →  latency (delay)
    stddev_rtt_ms            →  latency.jitter
    packet_loss (%)          →  (optional) timeout toxic
"""

import json
import argparse
import random
import math
import ast
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class NetworkProfile:
    """A single client's network profile derived from MobiPerf traces."""
    client_id: int
    # Bandwidth
    download_kbps: float
    upload_kbps: float
    # Latency
    latency_ms: float        # base RTT (one-way ≈ mean_rtt / 2)
    jitter_ms: float         # stddev of RTT
    # Derived
    network_type: str        # WIFI, LTE, MOBILE, etc.
    source_device: str       # original device from trace


@dataclass
class ToxiProxyConfig:
    """ToxiProxy configuration for one FL client."""
    client_id: int
    proxy_name: str
    listen_port: int
    upstream_port: int
    toxics: list


def parse_mobiperf_trace(trace_path: str) -> dict:
    """
    Parse MobiPerf JSON trace into structured measurements.
    
    Returns dict with keys: tcp_down, tcp_up, ping, http
    Each is a list of measurement dicts with network/device metadata.
    """
    with open(trace_path) as f:
        data = json.load(f)

    result = {"tcp_down": [], "tcp_up": [], "ping": [], "http": []}

    for record in data:
        if not record.get("success"):
            continue

        test_type = record.get("parameters", {}).get("type", "")
        dp = record.get("device_properties", {})
        di = dp.get("device_info", {}) or {}
        net_type = dp.get("network_type", "unknown")
        carrier = dp.get("carrier", "unknown")
        device = f"{di.get('manufacturer', '?')} {di.get('model', '?')}"

        if test_type == "tcpthroughput":
            raw_speeds = record.get("values", {}).get("tcp_speed_results", [])
            if isinstance(raw_speeds, str):
                try:
                    raw_speeds = ast.literal_eval(raw_speeds)
                except (ValueError, SyntaxError):
                    continue
            if not raw_speeds:
                continue

            speeds = [float(s) for s in raw_speeds]
            avg_kbps = sum(speeds) / len(speeds)
            direction = "up" if record.get("parameters", {}).get("dir_up") else "down"
            entry = {
                "avg_kbps": avg_kbps,
                "min_kbps": min(speeds),
                "max_kbps": max(speeds),
                "network_type": net_type,
                "device": device,
                "carrier": carrier,
            }
            if direction == "up":
                result["tcp_up"].append(entry)
            else:
                result["tcp_down"].append(entry)

        elif test_type == "ping":
            vals = record.get("values", {})
            mean_rtt = float(vals.get("mean_rtt_ms", 0))
            if mean_rtt <= 0:
                continue
            result["ping"].append({
                "mean_rtt_ms": mean_rtt,
                "min_rtt_ms": float(vals.get("min_rtt_ms", 0)),
                "max_rtt_ms": float(vals.get("max_rtt_ms", 0)),
                "stddev_rtt_ms": float(vals.get("stddev_rtt_ms", 0)),
                "packet_loss": float(vals.get("packet_loss", 0)),
                "network_type": net_type,
                "device": device,
                "carrier": carrier,
            })

    return result


def build_empirical_cdf(values: List[float]) -> List[float]:
    """Return sorted values for inverse CDF sampling."""
    return sorted(values)


def sample_from_cdf(sorted_values: List[float]) -> float:
    """Sample one value from the empirical distribution."""
    idx = random.randint(0, len(sorted_values) - 1)
    return sorted_values[idx]


def generate_client_profiles(
    measurements: dict,
    num_clients: int,
    seed: int = 42,
) -> List[NetworkProfile]:
    """
    Generate N client network profiles by sampling from the empirical
    distributions of MobiPerf measurements.
    
    Strategy:
    - For each client, independently sample download bandwidth, upload
      bandwidth, latency, and jitter from their respective distributions.
    - Optionally correlate: clients with low bandwidth tend to have
      higher latency (sample from the same quantile range).
    """
    random.seed(seed)

    # Build empirical distributions
    down_speeds = build_empirical_cdf(
        [m["avg_kbps"] for m in measurements["tcp_down"]]
    )
    up_speeds = build_empirical_cdf(
        [m["avg_kbps"] for m in measurements["tcp_up"]]
    )
    latencies = build_empirical_cdf(
        [m["mean_rtt_ms"] for m in measurements["ping"]]
    )
    jitters = build_empirical_cdf(
        [m["stddev_rtt_ms"] for m in measurements["ping"]]
    )

    # Collect network types and devices for metadata
    all_records = measurements["ping"] + measurements["tcp_down"]
    net_types = [r["network_type"] for r in all_records]
    devices = [r["device"] for r in all_records]

    profiles = []
    for i in range(num_clients):
        # Correlated sampling: use same quantile position for bandwidth & latency
        # Low bandwidth clients tend to have higher latency
        quantile = random.random()
        
        # Bandwidth: sample at quantile position
        down_idx = int(quantile * (len(down_speeds) - 1))
        up_idx = int(quantile * (len(up_speeds) - 1))
        
        # Latency: inverse correlation - low bandwidth → high latency
        lat_idx = int((1.0 - quantile) * (len(latencies) - 1))
        jit_idx = int((1.0 - quantile) * (len(jitters) - 1))

        # Add some noise (±20%) to avoid exact duplication
        noise = lambda v: v * random.uniform(0.8, 1.2)

        profile = NetworkProfile(
            client_id=i,
            download_kbps=max(10, noise(down_speeds[down_idx])),
            upload_kbps=max(10, noise(up_speeds[up_idx])),
            latency_ms=max(1, noise(latencies[lat_idx])),
            jitter_ms=max(0, noise(jitters[jit_idx])),
            network_type=random.choice(net_types),
            source_device=random.choice(devices),
        )
        profiles.append(profile)

    return profiles


def profile_to_toxiproxy_config(
    profile: NetworkProfile,
    base_listen_port: int = 18000,
    upstream_host: str = "localhost",
    upstream_port: int = 8080,
) -> ToxiProxyConfig:
    """
    Convert a NetworkProfile into ToxiProxy API configuration.
    
    ToxiProxy toxic types used:
    - bandwidth: limits throughput (in KB/s, so we convert from kbps)
    - latency: adds delay with jitter
    
    Note: ToxiProxy bandwidth rate is in KB/s (bytes, not bits).
    MobiPerf reports in kbps (kilobits per second).
    Conversion: KB/s = kbps / 8
    """
    listen_port = base_listen_port + profile.client_id

    toxics = []

    # --- Downstream bandwidth limit (server → client, i.e. model download) ---
    download_KBps = max(1, profile.download_kbps / 8.0)
    toxics.append({
        "name": f"bw_downstream_{profile.client_id}",
        "type": "bandwidth",
        "stream": "downstream",
        "attributes": {
            "rate": int(download_KBps),  # KB/s
        },
    })

    # --- Upstream bandwidth limit (client → server, i.e. model upload) ---
    upload_KBps = max(1, profile.upload_kbps / 8.0)
    toxics.append({
        "name": f"bw_upstream_{profile.client_id}",
        "type": "bandwidth",
        "stream": "upstream",
        "attributes": {
            "rate": int(upload_KBps),  # KB/s
        },
    })

    # --- Latency (applied to downstream; simulates network RTT) ---
    # ToxiProxy latency is one-way; RTT = 2 × one-way
    one_way_latency = max(1, int(profile.latency_ms / 2))
    one_way_jitter = max(0, int(profile.jitter_ms / 2))
    toxics.append({
        "name": f"latency_downstream_{profile.client_id}",
        "type": "latency",
        "stream": "downstream",
        "attributes": {
            "latency": one_way_latency,   # ms
            "jitter": one_way_jitter,      # ms
        },
    })

    # --- Latency on upstream too (for symmetric delay) ---
    toxics.append({
        "name": f"latency_upstream_{profile.client_id}",
        "type": "latency",
        "stream": "upstream",
        "attributes": {
            "latency": one_way_latency,
            "jitter": one_way_jitter,
        },
    })

    return ToxiProxyConfig(
        client_id=profile.client_id,
        proxy_name=f"fl_client_{profile.client_id}",
        listen_port=listen_port,
        upstream_port=upstream_port,
        toxics=toxics,
    )


def generate_toxiproxy_setup_script(configs: List[ToxiProxyConfig], upstream_host: str = "localhost") -> str:
    """Generate a bash script to set up all ToxiProxy proxies and toxics."""
    lines = [
        "#!/bin/bash",
        "# Auto-generated ToxiProxy setup for FL experiment",
        "# Requires: toxiproxy-server running on localhost:8474",
        "",
        "TOXIPROXY_URL=http://localhost:8474",
        "",
        "# Remove existing proxies",
    ]

    for cfg in configs:
        lines.append(f'curl -s -X DELETE $TOXIPROXY_URL/proxies/{cfg.proxy_name} 2>/dev/null')
    
    lines.append("")
    lines.append("# Create proxies and toxics")

    for cfg in configs:
        lines.append(f"")
        lines.append(f"# --- Client {cfg.client_id} ---")
        # Create proxy
        proxy_json = json.dumps({
            "name": cfg.proxy_name,
            "listen": f"0.0.0.0:{cfg.listen_port}",
            "upstream": f"{upstream_host}:{cfg.upstream_port}",
        })
        lines.append(
            f"curl -s -X POST $TOXIPROXY_URL/proxies "
            f"-H 'Content-Type: application/json' "
            f"-d '{proxy_json}'"
        )
        # Add toxics
        for toxic in cfg.toxics:
            toxic_json = json.dumps(toxic)
            lines.append(
                f"curl -s -X POST $TOXIPROXY_URL/proxies/{cfg.proxy_name}/toxics "
                f"-H 'Content-Type: application/json' "
                f"-d '{toxic_json}'"
            )

    lines.append("")
    lines.append("echo 'ToxiProxy setup complete.'")
    lines.append(f"echo 'Clients listen on ports {configs[0].listen_port}-{configs[-1].listen_port}'")
    return "\n".join(lines)


def print_summary(profiles: List[NetworkProfile]):
    """Print a summary table of generated client profiles."""
    print(f"\n{'='*90}")
    print(f"  Generated {len(profiles)} FL Client Network Profiles from MobiPerf Traces")
    print(f"{'='*90}")
    print(f"{'Client':>7} {'DL (kbps)':>12} {'UL (kbps)':>12} {'RTT (ms)':>10} {'Jitter (ms)':>12} {'Network':>8}")
    print(f"{'-'*7:>7} {'-'*12:>12} {'-'*12:>12} {'-'*10:>10} {'-'*12:>12} {'-'*8:>8}")
    for p in profiles:
        print(f"{p.client_id:>7} {p.download_kbps:>12.1f} {p.upload_kbps:>12.1f} "
              f"{p.latency_ms:>10.1f} {p.jitter_ms:>12.1f} {p.network_type:>8}")

    dls = [p.download_kbps for p in profiles]
    uls = [p.upload_kbps for p in profiles]
    lats = [p.latency_ms for p in profiles]
    print(f"\n  Download: {min(dls):.0f} - {max(dls):.0f} kbps (median {sorted(dls)[len(dls)//2]:.0f})")
    print(f"  Upload:   {min(uls):.0f} - {max(uls):.0f} kbps (median {sorted(uls)[len(uls)//2]:.0f})")
    print(f"  Latency:  {min(lats):.0f} - {max(lats):.0f} ms (median {sorted(lats)[len(lats)//2]:.0f})")
    print(f"  Heterogeneity ratio (DL): {max(dls)/max(1,min(dls)):.1f}x")
    print(f"  Heterogeneity ratio (RTT): {max(lats)/max(1,min(lats)):.1f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ToxiProxy configs from MobiPerf traces for FL experiments"
    )
    parser.add_argument("--trace", required=True, help="Path to MobiPerf JSON trace file")
    parser.add_argument("--num_clients", type=int, default=20, help="Number of FL clients")
    parser.add_argument("--output", default="toxiproxy_configs.json", help="Output config file")
    parser.add_argument("--script", default="setup_toxiproxy.sh", help="Output bash setup script")
    parser.add_argument("--upstream_host", default="localhost", help="FL server host")
    parser.add_argument("--upstream_port", type=int, default=8080, help="FL server port")
    parser.add_argument("--base_port", type=int, default=18000, help="Starting port for ToxiProxy listeners")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Parse trace
    print(f"Parsing MobiPerf trace: {args.trace}")
    measurements = parse_mobiperf_trace(args.trace)
    print(f"  TCP download: {len(measurements['tcp_down'])} measurements")
    print(f"  TCP upload:   {len(measurements['tcp_up'])} measurements")
    print(f"  Ping:         {len(measurements['ping'])} measurements")

    if not measurements["tcp_down"] or not measurements["ping"]:
        print("ERROR: Insufficient measurements. Need both TCP throughput and ping data.")
        sys.exit(1)

    # Generate profiles
    profiles = generate_client_profiles(measurements, args.num_clients, seed=args.seed)
    print_summary(profiles)

    # Convert to ToxiProxy configs
    configs = [
        profile_to_toxiproxy_config(p, args.base_port, args.upstream_host, args.upstream_port)
        for p in profiles
    ]

    # Save JSON config
    output_data = {
        "metadata": {
            "trace_file": args.trace,
            "num_clients": args.num_clients,
            "seed": args.seed,
            "upstream": f"{args.upstream_host}:{args.upstream_port}",
        },
        "profiles": [asdict(p) for p in profiles],
        "toxiproxy_configs": [asdict(c) for c in configs],
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved ToxiProxy configs: {args.output}")

    # Save bash script
    script = generate_toxiproxy_setup_script(configs, args.upstream_host)
    with open(args.script, "w") as f:
        f.write(script)
    print(f"Saved setup script: {args.script}")


if __name__ == "__main__":
    main()