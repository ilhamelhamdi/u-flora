#!/usr/bin/env python3
"""Experiment orchestrator.

Manages the Flower federation lifecycle on Apptainer:
  - Generates and assigns heterogeneous device profiles to clients
  - Starts superlink + supernodes with profile-aware configuration
  - Runs single or batched experiments
  - Manages ToxiProxy for network heterogeneity

Commands:
    python setup.py up                      # Start infrastructure
    python setup.py run   task=text_classification model=distilbert dataset=sst2
    python setup.py batch --batch-config experiments.yaml
    python setup.py down                    # Cleanup
    python setup.py profiles --show         # Inspect generated profiles

Architecture:
    ┌──────────┐
    │ SuperLink│<───────────┐
    │ :54001-3 │            │
    └────┬─────┘            │
         │                  │
   ┌─────┴──────┐           │
   │  ToxiProxy │           │
   │  :18000+N  │           │ (no proxy)
   └────────────┘           │
         │ (via ToxiProxy)  │
         │                  │
    ┌────┴────┐             │
    │SuperNode│<────────────┘
    │  :54100 │ (proxy) 
    ├─────────┤
    │SuperNode│   Each node has:
    │  :54101 │   - Device profile (compute + network)
    ├─────────┤   - Injected compute delay
    │  ...    │   - ToxiProxy traffic shaping
    └─────────┘
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("u_flora.setup")

# ── Paths & Ports ─────────────────────────────────────────────────────────────

SIF_FILE = "flwr.sif"
LOG_DIR = "logs"
PROFILE_DIR = "device_profiles"
NETWORK_PROFILE_TRACE_FILE = str(Path(__file__).parent / "traces" / "network" / "trace.json")
COMPUTE_PROFILE_TRACE_FILE = str(Path(__file__).parent / "traces" / "computation" / "client_device_capacity.json")

TOXIPROXY_API_PORT = 8474 # Default ToxiProxy API port
SUPERLINK_PORTS = {
    "serverappio": 15001,
    "fleet": 15002,
    "control": 15003,
}
SUPERNODE_PORT_START = 15100
TOXIPROXY_PROXY_PORT_START = 16100

# Note:
# Check the availability of these ports before running, or adjust as needed to avoid conflicts
# Command to check:
# `ss -tlnp | awk 'NR>1 {print $4}' | grep -oP '(?<=:)\d+' | awk -v lo=15000 -v hi=15200 '$1>=lo && $1<=hi' | sort -n`
# Change `lo` and `hi` to check different ranges.

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclasses.dataclass
class SupernodeSpec:
    """Declarative description of one supernode instance."""
    name: str
    node_id: int
    profile: dict
    clientappio_port: int
    superlink_address: str  # "127.0.0.1:{18000+N}" (via ToxiProxy) or "127.0.0.1:54002"
    total_nodes: int

def get_node_name(node_id: int) -> str:
    return f"supernode-{node_id:03d}"

# ── Main CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="U-Flora experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- up ---
    p_up = sub.add_parser("up", help="Start superlink + supernodes")
    p_up.add_argument("-n", "--num-clients", type=int, default=20)
    p_up.add_argument("--network-trace", type=str, default=NETWORK_PROFILE_TRACE_FILE)
    p_up.add_argument("--compute-trace", type=str, default=COMPUTE_PROFILE_TRACE_FILE)
    p_up.add_argument("--seed", type=int, default=42)
    p_up.add_argument("--no-toxiproxy", action="store_true")

    # --- down ---
    sub.add_parser("down", help="Stop all instances")

    # --- run ---
    p_run = sub.add_parser("run", help="Run single experiment")
    p_run.add_argument("overrides", nargs="*", help="Hydra-style overrides")

    # --- batch ---
    p_batch = sub.add_parser("batch", help="Run batch experiments")
    p_batch.add_argument("--batch-config", required=True, type=str)

   
    # --- profiles ---
    p_prof = sub.add_parser("profiles", help="Generate/inspect device profiles")
    p_prof.add_argument("-n", "--num-clients", type=int, default=100)
    p_prof.add_argument("--network-trace", type=str, default=NETWORK_PROFILE_TRACE_FILE)
    p_prof.add_argument("--compute-trace", type=str, default=COMPUTE_PROFILE_TRACE_FILE)
    p_prof.add_argument("--seed", type=int, default=42)
    p_prof.add_argument("--show", action="store_true", help="Print summary")

    args = parser.parse_args()

    if args.command == "up":
        logger.debug("Parsed args: %s", args)
        up(
            num_clients=args.num_clients,
            network_trace=args.network_trace,
            compute_trace=args.compute_trace,
            seed=args.seed,
            use_toxiproxy=not args.no_toxiproxy,
        )
    elif args.command == "down":
        down()
    elif args.command == "run":
        run_task(args.overrides)
    elif args.command == "batch":
        run_batch(args.batch_config)
    elif args.command == "profiles":
        profiles = generate_profiles(
            args.num_clients, args.network_trace, args.compute_trace, args.seed
        )
        if args.show:
            _print_profile_summary(profiles)


# ── Apptainer Instance Management ────────────────────────────────────────────

# ── Up ───────────────────────────────────────────────────────────────────────


def up(
    num_clients: int = 20,
    network_trace: str | None = None,
    compute_trace: str | None = None,
    seed: int = 42,
    use_toxiproxy: bool = True,
) -> None:
    """Start superlink, configure ToxiProxy, and spawn supernodes."""
    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(SIF_FILE):
        logger.error("%s not found. Build with 'apptainer pull' first.", SIF_FILE)
        return

    # Phase 1 — Describe: generate profiles and build specs
    logger.info("Generating %d device profiles...", num_clients)
    profiles = generate_profiles(num_clients, network_trace, compute_trace, seed)

    def _superlink_addr(node_id: int) -> str:
        if use_toxiproxy:
            return f"127.0.0.1:{TOXIPROXY_PROXY_PORT_START + node_id}"
        return f"127.0.0.1:{SUPERLINK_PORTS['fleet']}"

    specs = [
        SupernodeSpec(
            name=get_node_name(p["client_id"]),
            node_id=p["client_id"],
            profile=p,
            clientappio_port=SUPERNODE_PORT_START + p["client_id"],
            superlink_address=_superlink_addr(p["client_id"]),
            total_nodes=len(profiles),
        )
        for p in profiles
    ]

    # Phase 2 — Start superlink and wait for it to be reachable
    logger.info("Starting Superlink...")
    _launch_superlink()
    logger.info("Waiting for Superlink to be ready...")
    _wait_for_superlink()

    # Phase 3 — Configure ToxiProxy
    if use_toxiproxy:
        try:
            _configure_all_toxiproxy(profiles)
        except Exception as e:
            logger.error(
                "ToxiProxy configuration failed: %s — aborting to avoid running "
                "experiments without network simulation. Pass --no-toxiproxy to "
                "explicitly opt out.",
                e,
            )
            return

    # Phase 4 — Launch supernodes in parallel
    logger.info("Spawning %d supernodes...", len(specs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(_launch_supernode, spec): spec for spec in specs}
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            if exc := future.exception():
                logger.error("Failed to start supernode-%d: %s", spec.node_id, exc)

    logger.info("All %d supernodes started.", len(specs))
    subprocess.run(["apptainer", "instance", "list"], check=False)


def _build_superlink_cmd() -> list[str]:
    return [
        "apptainer",
        "exec",
        "instance://superlink",
        "flower-superlink",
        "--insecure",
        "--isolation",
        "subprocess",
        "--serverappio-api-address",
        f"0.0.0.0:{SUPERLINK_PORTS['serverappio']}",
        "--fleet-api-address",
        f"0.0.0.0:{SUPERLINK_PORTS['fleet']}",
        "--control-api-address",
        f"0.0.0.0:{SUPERLINK_PORTS['control']}",
    ]


def _launch_superlink(log_dir: str = LOG_DIR) -> subprocess.Popen:
    subprocess.run(
        ["apptainer", "instance", "start", SIF_FILE, "superlink"],
        check=False,
    )

    proc = subprocess.Popen(
        _build_superlink_cmd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def _filter_log() -> None:
        """Filter out heartbeat logs from superlink"""
        with open(Path(log_dir) / "superlink.log", "w") as fh:
            for line in iter(proc.stdout.readline, ""):
                if "Fleet.PullMessages" not in line:
                    fh.write(line)
                    fh.flush()

    threading.Thread(target=_filter_log, daemon=True).start()
    return proc


def _build_supernode_cmd(spec: SupernodeSpec) -> list[str]:
    return [
        "apptainer",
        "exec",
        "--env",
        f"DEVICE_PROFILE_PATH={PROFILE_DIR}/{spec.name}.json",
        f"instance://{spec.name}",
        "flower-supernode",
        "--insecure",
        "--superlink",
        spec.superlink_address,
        "--clientappio-api-address",
        f"0.0.0.0:{spec.clientappio_port}",
        "--isolation",
        "subprocess",
        "--node-config",
        f"partition-id={spec.node_id} num-partitions={spec.total_nodes}",
    ]


def _launch_supernode(spec: SupernodeSpec, log_dir: str = LOG_DIR) -> subprocess.Popen:
    instance_name = spec.name
    subprocess.run(
        ["apptainer", "instance", "start", SIF_FILE, instance_name],
        check=False,
    )

    log_path = Path(log_dir) / f"{instance_name}.log"
    log_fh = open(log_path, "w")  # noqa: SIM115 — owned by subprocess until down()
    proc = subprocess.Popen(
        _build_supernode_cmd(spec),
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )
    logger.debug(
        "Started %s on port %d via %s",
        instance_name,
        spec.clientappio_port,
        spec.superlink_address,
    )
    return proc


def _wait_for_superlink(
    host: str = "127.0.0.1",
    port: int | None = None,
    timeout_s: int = 30,
    poll_interval_s: float = 0.5,
) -> None:
    """Poll until the superlink fleet port is reachable or raise on timeout."""
    port = port or SUPERLINK_PORTS["fleet"]
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Superlink did not become reachable at {host}:{port} "
                    f"within {timeout_s}s"
                )
            time.sleep(poll_interval_s)


# ── Down ───────────────────────────────────────────────────────────────────────


def down() -> None:
    """Stop all Apptainer instances and remove ToxiProxy proxies."""
    logger.info("Stopping all Flower instances...")
    subprocess.run(["apptainer", "instance", "stop", "--all"], check=False)
    _teardown_toxiproxy()
    logger.info("Done.")


def _teardown_toxiproxy() -> None:
    """Delete all supernode-* proxies from ToxiProxy (best-effort)."""
    api_base = f"http://localhost:{TOXIPROXY_API_PORT}"
    try:
        resp = requests.get(f"{api_base}/proxies", timeout=3)
        resp.raise_for_status()
    except Exception:
        logger.debug("ToxiProxy not reachable during teardown — skipping.")
        return

    proxies = resp.json()
    fl_proxies = [name for name in proxies if name.startswith("supernode-")]

    if not fl_proxies:
        return

    for name in fl_proxies:
        try:
            requests.delete(f"{api_base}/proxies/{name}", timeout=5)
        except Exception as e:
            logger.warning("Failed to delete ToxiProxy proxy %s: %s", name, e)

    logger.info("Removed %d ToxiProxy proxies.", len(fl_proxies))


# ── Experiment Execution ──────────────────────────────────────────────────────

def run_task(overrides: list[str]) -> None:
    """Run a single FL experiment via ``flwr run``.

    Example:
        run_task(["task=text_classification", "model=distilbert", "dataset=sst2",
                  "strategy=oort", "num_server_rounds=100"])
    """
    cmd = [
        "flwr",
        "run",
        ".",
        "--insecure",
        "--serverappio-api-address",
        f"127.0.0.1:{SUPERLINK_PORTS['serverappio']}",
    ]

    for ov in overrides:
        cmd.extend(["--run-config", ov])

    logger.info("Running: %s", " ".join(cmd))

    log_path = f"{LOG_DIR}/flower-task.log"
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=log_file)

    if result.returncode == 0:
        logger.info("Task completed successfully. Logs: %s", log_path)
    else:
        logger.error("Task failed (rc=%d). Check %s", result.returncode, log_path)


def run_batch(batch_config_path: str) -> None:
    """Run a batch of experiments sequentially.

    The batch config is a YAML file listing experiments:

    ```yaml
    experiments:
      - name: "random_sst2"
        overrides:
          - "strategy=random"
          - "dataset=sst2"
          - "num_server_rounds=100"
      - name: "oort_sst2"
        overrides:
          - "strategy=oort"
          - "dataset=sst2"
          - "num_server_rounds=100"
    ```
    """
    import yaml

    with open(batch_config_path) as f:
        batch = yaml.safe_load(f)

    experiments = batch.get("experiments", [])
    total = len(experiments)

    for idx, exp in enumerate(experiments, 1):
        name = exp.get("name", f"experiment_{idx}")
        overrides = exp.get("overrides", [])
        logger.info("=" * 60)
        logger.info("Batch [%d/%d]: %s", idx, total, name)
        logger.info("=" * 60)

        overrides.append(f"wandb.run_name={name}")
        run_task(overrides)

        time.sleep(5)

    logger.info("Batch complete: %d/%d experiments run.", total, total)


# ── Device Profile Management ─────────────────────────────────────────────────


def generate_profiles(
    num_clients: int,
    network_trace: str | None = None,
    compute_trace: str | None = None,
    seed: int = 42,
    output_dir: str = PROFILE_DIR,
) -> list[dict]:
    """Generate and persist device profiles from trace data.

    Each profile is saved as a JSON file for reproducibility and inspection.
    Returns the list of profile dicts.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from u_flora.selection.profile_generator import generate_device_profiles

    profiles = generate_device_profiles(
        num_profiles=num_clients,
        network_trace_path=network_trace,
        compute_trace_path=compute_trace,
        seed=seed,
    )

    os.makedirs(output_dir, exist_ok=True)

    profile_dicts = []
    for p in profiles:
        d = {
            "client_id": p.client_id,
            "computation_latency_ms": p.computation_latency_ms,
            "download_kbps": p.download_kbps,
            "upload_kbps": p.upload_kbps,
            "latency_ms": p.latency_ms,
            "jitter_ms": p.jitter_ms,
            "network_type": p.network_type,
            "device_name": p.device_name,
        }
        profile_dicts.append(d)

    combined_path = os.path.join(output_dir, "all_profiles.json")
    with open(combined_path, "w") as f:
        json.dump(profile_dicts, f, indent=2)
    logger.info("Saved %d profiles to %s", len(profile_dicts), combined_path)

    for d in profile_dicts:
        path = os.path.join(output_dir, f"{get_node_name(d['client_id'])}.json")
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    return profile_dicts


def load_profiles(profile_dir: str = PROFILE_DIR) -> list[dict]:
    """Load previously generated profiles."""
    combined = os.path.join(profile_dir, "all_profiles.json")
    if not os.path.exists(combined):
        logger.error(
            "No profiles found at %s. Run 'setup.py profiles' first.", combined
        )
        return []
    with open(combined) as f:
        return json.load(f)


# ── ToxiProxy Setup ───────────────────────────────────────────────────────────


def _configure_all_toxiproxy(
    profiles: list[dict],
    upstream_host: str = "127.0.0.1",
    upstream_port: int | None = None,
) -> None:
    """Configure ToxiProxy proxies for all clients.

    Raises RuntimeError if ToxiProxy is unreachable, or if any client
    configuration fails.
    """
    api_base = f"http://localhost:{TOXIPROXY_API_PORT}"

    try:
        requests.get(f"{api_base}/proxies", timeout=3)
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"ToxiProxy not reachable at {api_base}. "
            "Ensure toxiproxy-server is running, or pass --no-toxiproxy."
        ) from e

    failures: list[str] = []
    for p in profiles:
        proxy_port = TOXIPROXY_PROXY_PORT_START + p["client_id"]
        try:
            _configure_toxiproxy_for_client(
                api_base, p, proxy_port, upstream_host, upstream_port
            )
        except Exception as e:
            failures.append(f"client_{p['client_id']}: {e}")

    if failures:
        raise RuntimeError(
            f"ToxiProxy configuration failed for {len(failures)} client(s):\n"
            + "\n".join(failures)
        )

    logger.info(
        "ToxiProxy configured for %d clients (ports %d–%d)",
        len(profiles),
        TOXIPROXY_PROXY_PORT_START,
        TOXIPROXY_PROXY_PORT_START + len(profiles) - 1,
    )


def _configure_toxiproxy_for_client(
    api_base: str,
    profile: dict,
    proxy_port: int,
    upstream_host: str = "127.0.0.1",
    upstream_port: int | None = None,
) -> None:
    """Configure one ToxiProxy proxy with bandwidth and latency toxics."""
    upstream_port = upstream_port or SUPERLINK_PORTS["fleet"]
    cid = profile["client_id"]
    proxy_name = get_node_name(cid)

    requests.delete(f"{api_base}/proxies/{proxy_name}", timeout=5)

    resp = requests.post(
        f"{api_base}/proxies",
        json={
            "name": proxy_name,
            "listen": f"0.0.0.0:{proxy_port}",
            "upstream": f"{upstream_host}:{upstream_port}",
        },
        timeout=5,
    )
    resp.raise_for_status()

    toxics = [
        # Bandwidth (downstream) — KB/s = kbps / 8
        (
            f"bw_down_{cid}",
            "bandwidth",
            "downstream",
            {"rate": max(1, int(profile["download_kbps"] / 8))},
        ),
        # Bandwidth (upstream)
        (
            f"bw_up_{cid}",
            "bandwidth",
            "upstream",
            {"rate": max(1, int(profile["upload_kbps"] / 8))},
        ),
        # Latency — half RTT each way
        (
            f"lat_down_{cid}",
            "latency",
            "downstream",
            {
                "latency": max(1, int(profile["latency_ms"] / 2)),
                "jitter": max(0, int(profile["jitter_ms"] / 2)),
            },
        ),
        (
            f"lat_up_{cid}",
            "latency",
            "upstream",
            {
                "latency": max(1, int(profile["latency_ms"] / 2)),
                "jitter": max(0, int(profile["jitter_ms"] / 2)),
            },
        ),
    ]

    for name, toxic_type, stream, attributes in toxics:
        resp = requests.post(
            f"{api_base}/proxies/{proxy_name}/toxics",
            json={
                "name": name,
                "type": toxic_type,
                "stream": stream,
                "attributes": attributes,
            },
            timeout=5,
        )
        resp.raise_for_status()


# ── Utilities ─────────────────────────────────────────────────────────────────


def _print_profile_summary(profiles: list[dict]) -> None:
    """Print a concise summary of the generated profiles."""
    n = len(profiles)
    comp = sorted(p["computation_latency_ms"] for p in profiles)
    dl = sorted(p["download_kbps"] for p in profiles)
    rtt = sorted(p["latency_ms"] for p in profiles)
    net_types: dict[str, int] = {}
    for p in profiles:
        nt = p["network_type"]
        net_types[nt] = net_types.get(nt, 0) + 1

    def percentile(vals: list, pct: int) -> float:
        return vals[min(int(len(vals) * pct / 100), len(vals) - 1)]

    print(f"\n{'=' * 70}")
    print(f"  {n} Device Profiles")
    print(f"{'=' * 70}")
    print(
        f"  {'Metric':<25} {'P10':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P90':>10}"
    )
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for label, vals in [
        ("Compute (ms/sample)", comp),
        ("Download (kbps)", dl),
        ("RTT (ms)", rtt),
    ]:
        row = [f"{percentile(vals, p):.1f}" for p in [10, 25, 50, 75, 90]]
        print(f"  {label:<25} {'':>3}".rstrip() + "  ".join(f"{v:>10}" for v in row))

    print(f"\n  Network types: {net_types}")
    print(f"  Heterogeneity (compute): {comp[-1] / max(comp[0], 0.01):.1f}x")
    print(f"  Heterogeneity (network): {dl[-1] / max(dl[0], 0.01):.1f}x")


if __name__ == "__main__":
    main()
