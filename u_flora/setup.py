#!/usr/bin/env python3
"""U-Flora experiment orchestrator.

Manages the Flower federation lifecycle on HPC (Apptainer) or local (Docker):
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

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("u_flora.setup")

# -- Paths & Ports -------------------------------------------------------------

SIF_FILE = "flwr.sif"
LOG_DIR = "logs"
PROFILE_DIR = "device_profiles"

SUPERLINK_PORTS = {
    "serverappio": 54001,
    "fleet": 54002,
    "control": 54003,
}
SUPERNODE_PORT_START = 54100
TOXIPROXY_API_PORT = 8474
TOXIPROXY_PROXY_PORT_START = 18000


# -- Device Profile Management -------------------------------------------------

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
    # Import here to avoid requiring u_flora package for basic setup operations
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

    # Save combined file
    combined_path = os.path.join(output_dir, "all_profiles.json")
    with open(combined_path, "w") as f:
        json.dump(profile_dicts, f, indent=2)
    logger.info("Saved %d profiles to %s", len(profile_dicts), combined_path)

    # Also save per-client files for supernodes to read
    for d in profile_dicts:
        path = os.path.join(output_dir, f"client_{d['client_id']}.json")
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    return profile_dicts


def load_profiles(profile_dir: str = PROFILE_DIR) -> list[dict]:
    """Load previously generated profiles."""
    combined = os.path.join(profile_dir, "all_profiles.json")
    if not os.path.exists(combined):
        logger.error(
            "No profiles found at %s. Run 'setup.py profiles' first.", combined)
        return []
    with open(combined) as f:
        return json.load(f)


# -- ToxiProxy Setup -----------------------------------------------------------

def setup_toxiproxy(
    profiles: list[dict],
    upstream_host: str = "127.0.0.1",
    upstream_port: int | None = None,
) -> None:
    """Configure ToxiProxy proxies for network heterogeneity.

    Creates one proxy per client with bandwidth and latency toxics
    derived from the client's network profile.

    Assumes toxiproxy-server is already running on localhost:8474.
    """
    if upstream_port is None:
        upstream_port = SUPERLINK_PORTS["fleet"]

    api = f"http://localhost:{TOXIPROXY_API_PORT}"

    for p in profiles:
        cid = p["client_id"]
        proxy_name = f"fl_client_{cid}"
        listen_port = TOXIPROXY_PROXY_PORT_START + cid

        # Delete existing proxy (ignore errors)
        subprocess.run(
            ["curl", "-s", "-X", "DELETE", f"{api}/proxies/{proxy_name}"],
            capture_output=True,
        )

        # Create proxy
        proxy_cfg = json.dumps({
            "name": proxy_name,
            "listen": f"0.0.0.0:{listen_port}",
            "upstream": f"{upstream_host}:{upstream_port}",
        })
        subprocess.run(
            ["curl", "-s", "-X", "POST", f"{api}/proxies",
             "-H", "Content-Type: application/json", "-d", proxy_cfg],
            capture_output=True,
        )

        # Bandwidth toxic (downstream) — KB/s = kbps / 8
        dl_rate = max(1, int(p["download_kbps"] / 8))
        _add_toxic(api, proxy_name, f"bw_down_{cid}", "bandwidth",
                   "downstream", {"rate": dl_rate})

        # Bandwidth toxic (upstream)
        ul_rate = max(1, int(p["upload_kbps"] / 8))
        _add_toxic(api, proxy_name, f"bw_up_{cid}", "bandwidth",
                   "upstream", {"rate": ul_rate})

        # Latency toxic (both directions) — half RTT each way
        half_rtt = max(1, int(p["latency_ms"] / 2))
        half_jitter = max(0, int(p["jitter_ms"] / 2))
        _add_toxic(api, proxy_name, f"lat_down_{cid}", "latency",
                   "downstream", {"latency": half_rtt, "jitter": half_jitter})
        _add_toxic(api, proxy_name, f"lat_up_{cid}", "latency",
                   "upstream", {"latency": half_rtt, "jitter": half_jitter})

    logger.info(
        "ToxiProxy configured for %d clients (ports %d-%d)",
        len(profiles),
        TOXIPROXY_PROXY_PORT_START,
        TOXIPROXY_PROXY_PORT_START + len(profiles) - 1,
    )


def _add_toxic(
    api: str, proxy: str, name: str, toxic_type: str,
    stream: str, attributes: dict,
) -> None:
    toxic_cfg = json.dumps({
        "name": name, "type": toxic_type,
        "stream": stream, "attributes": attributes,
    })
    subprocess.run(
        ["curl", "-s", "-X", "POST",
         f"{api}/proxies/{proxy}/toxics",
         "-H", "Content-Type: application/json", "-d", toxic_cfg],
        capture_output=True,
    )


# -- Apptainer Instance Management --------------------------------------------

def up(
    num_clients: int = 20,
    network_trace: str | None = None,
    compute_trace: str | None = None,
    seed: int = 42,
    use_toxiproxy: bool = True,
) -> None:
    """Start superlink, generate profiles, configure ToxiProxy, spawn supernodes."""
    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(SIF_FILE):
        logger.error(
            "%s not found. Build with 'apptainer pull' first.", SIF_FILE)
        return

    # 1. Generate device profiles
    logger.info("Generating %d device profiles...", num_clients)
    profiles = generate_profiles(
        num_clients, network_trace, compute_trace, seed
    )

    # 2. Start Superlink
    logger.info("Starting Superlink...")
    subprocess.run(
        ["apptainer", "instance", "start", SIF_FILE, "superlink"],
        check=False,
    )

    link_log = open(f"{LOG_DIR}/superlink.log", "w")
    link_cmd = [
        "apptainer", "exec", "instance://superlink",
        "flower-superlink", "--insecure", "--isolation", "subprocess",
        "--serverappio-api-address", f"0.0.0.0:{SUPERLINK_PORTS['serverappio']}",
        "--fleet-api-address", f"0.0.0.0:{SUPERLINK_PORTS['fleet']}",
        "--control-api-address", f"0.0.0.0:{SUPERLINK_PORTS['control']}",
    ]

    proc = subprocess.Popen(
        link_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )

    def _filter_log():
        for line in iter(proc.stdout.readline, ""):
            if "Fleet.PullMessages" not in line:
                link_log.write(line)
                link_log.flush()

    threading.Thread(target=_filter_log, daemon=True).start()
    time.sleep(5)

    # 3. Configure ToxiProxy (if available)
    if use_toxiproxy:
        try:
            setup_toxiproxy(profiles)
        except Exception as e:
            logger.warning(
                "ToxiProxy setup failed: %s (continuing without)", e)

    # 4. Spawn Supernodes — one per client, each with its profile
    total = len(profiles)
    logger.info("Spawning %d supernodes...", total)

    for p in profiles:
        node_id = p["client_id"]
        instance_name = f"supernode-{node_id}"
        port = SUPERNODE_PORT_START + node_id

        subprocess.run(
            ["apptainer", "instance", "start", SIF_FILE, instance_name],
            check=False,
        )

        node_log = open(f"{LOG_DIR}/{instance_name}.log", "w")
        exec_cmd = [
            "apptainer", "exec",
            "--env", f"DEVICE_PROFILE_PATH={PROFILE_DIR}/client_{node_id}.json",
            f"instance://{instance_name}",
            "flower-supernode", "--insecure",
            "--superlink", f"127.0.0.1:{SUPERLINK_PORTS['fleet']}",
            "--clientappio-api-address", f"0.0.0.0:{port}",
            "--isolation", "subprocess",
            "--node-config",
            f"partition-id={node_id} num-partitions={total}",
        ]

        subprocess.Popen(
            exec_cmd, stdout=node_log, stderr=node_log,
            start_new_session=True,
        )
        logger.debug("Started %s on port %d", instance_name, port)

    logger.info("All %d supernodes started.", total)
    subprocess.run(["apptainer", "instance", "list"], check=False)


def down() -> None:
    """Stop all Apptainer instances."""
    logger.info("Stopping all Flower instances...")
    subprocess.run(["apptainer", "instance", "stop", "--all"], check=False)
    logger.info("Done.")


# -- Experiment Execution ------------------------------------------------------

def run_task(overrides: str | None) -> None:
    """Run a single FL experiment via ``flwr run``.

    Overrides are Hydra-style key=value pairs that get forwarded to
    the Flower run config.

    Example:
        run_task("task='text_classification' model='distilbert' dataset='sst2' strategy='oort',num_server_rounds=100")
    """
    # Build the flwr run command with overrides
    cmd = [
        "flwr", "run", "."
    ]

    # Forward overrides as Flower run-config
    if overrides:
        cmd.extend(["--run-config", overrides])

    logger.info("Running: %s", " ".join(cmd))

    log_path = f"{LOG_DIR}/flower-task.log"
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd, stdout=log_file, stderr=log_file,
        )

    if result.returncode == 0:
        logger.info("Task completed successfully. Logs: %s", log_path)
    else:
        logger.error("Task failed (rc=%d). Check %s",
                     result.returncode, log_path)


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

        # Add wandb run name
        overrides.append(f"wandb.run_name={name}")
        run_task(overrides)

        # Brief pause between experiments
        time.sleep(5)

    logger.info("Batch complete: %d/%d experiments run.", total, total)


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    root_dir = Path(__file__).resolve().parent.parent
    default_net_trace = str(root_dir / "traces" / "network" / "trace.json")
    default_comp_trace = str(root_dir / "traces" /
                             "computation" / "client_device_capacity.json")

    parser = argparse.ArgumentParser(
        description="U-Flora experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- up ---
    p_up = sub.add_parser("up", help="Start superlink + supernodes")
    p_up.add_argument("-n", "--num-clients", type=int, default=20)
    p_up.add_argument("--network-trace", type=str, default=default_net_trace)
    p_up.add_argument("--compute-trace", type=str, default=default_comp_trace)
    p_up.add_argument("--seed", type=int, default=42)
    p_up.add_argument("--no-toxiproxy", action="store_true")

    # --- down ---
    sub.add_parser("down", help="Stop all instances")

    # --- run ---
    p_run = sub.add_parser("run", help="Run single experiment")
    p_run.add_argument("--run-config", type=str, help="Overrides for flwr run config")

    # --- batch ---
    p_batch = sub.add_parser("batch", help="Run batch experiments")
    p_batch.add_argument("--batch-config", required=True, type=str)

    # --- profiles ---
    p_prof = sub.add_parser(
        "profiles", help="Generate/inspect device profiles")
    p_prof.add_argument("-n", "--num-clients", type=int, default=100)
    p_prof.add_argument("--network-trace", type=str, default=default_net_trace)
    p_prof.add_argument("--compute-trace", type=str,
                        default=default_comp_trace)
    p_prof.add_argument("--seed", type=int, default=42)
    p_prof.add_argument("--show", action="store_true", help="Print summary")

    args = parser.parse_args()

    if args.command == "up":
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
        run_task(args.run_config)
    elif args.command == "batch":
        run_batch(args.batch_config)
    elif args.command == "profiles":
        profiles = generate_profiles(
            args.num_clients, args.network_trace, args.compute_trace, args.seed
        )
        if args.show:
            _print_profile_summary(profiles)


def _print_profile_summary(profiles: list[dict]) -> None:
    """Print a concise summary of the generated profiles."""
    n = len(profiles)
    comp = sorted(p["computation_latency_ms"] for p in profiles)
    dl = sorted(p["download_kbps"] for p in profiles)
    ul = sorted(p["upload_kbps"] for p in profiles)
    rtt = sorted(p["latency_ms"] for p in profiles)
    net_types = {}
    for p in profiles:
        nt = p["network_type"]
        net_types[nt] = net_types.get(nt, 0) + 1

    def percentile(vals, pct):
        return vals[min(int(len(vals) * pct / 100), len(vals) - 1)]

    print(f"\n{'=' * 70}")
    print(f"  {n} Device Profiles")
    print(f"{'=' * 70}")
    print(
        f"  {'Metric':<25} {'P10':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P90':>10}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for label, vals in [
        ("Compute (ms/sample)", comp),
        ("Download (kbps)", dl),
        ("Upload (kbps)", ul),
        ("Latency (ms)", rtt),
    ]:
        row = [f"{percentile(vals, p):.1f}" for p in [10, 25, 50, 75, 90]]
        print(f"  {label:<25}" + "  ".join(f"{v:>10}" for v in row))

    print(f"\n  Network types: {net_types}")
    print(f"  Heterogeneity (compute): {comp[-1] / max(comp[0], 0.01):.1f}x")
    print(f"  Heterogeneity (network - download): {dl[-1] / max(dl[0], 0.01):.1f}x")
    print(f"  Heterogeneity (network - upload): {ul[-1] / max(ul[0], 0.01):.1f}x")


if __name__ == "__main__":
    main()
