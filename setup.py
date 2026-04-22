#!/usr/bin/env python3
"""Experiment orchestrator.

Manages the Flower federation lifecycle:
  - Generates and assigns heterogeneous device profiles to clients
  - Starts superlink + supernodes with profile-aware configuration
  - Runs single or batched experiments
  - Manages ToxiProxy for network heterogeneity

Commands:
    python setup.py up --namespace=default  # Start infrastructure
    python setup.py run   task=text_classification model=distilbert dataset=sst2
    python setup.py batch --batch-config experiments.yaml
    python setup.py down                    # Cleanup
    python setup.py profiles --namespace=moderate --beta=2.0 --show  # Sample scenario profiles

Architecture:
    ┌──────────┐
    │ SuperLink│<───────────┐
    │ :15001-3 │            │
    └────┬─────┘            │
         │                  │
   ┌─────┴──────┐           │
   │  ToxiProxy │           │ (no proxy)
   │  :16100+N  │           │
   └────────────┘           │
         │ (via ToxiProxy)  │
         │                  │
    ┌────┴────┐             │
    │SuperNode│<────────────┘
    │  :15100 │ (proxy)
    ├─────────┤
    │SuperNode│   Each node has:
    │  :15101 │   - Device profile (compute + network)
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
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger("u_flora.setup")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

# Paths
LOG_DIR = "logs"
LOG_DIR_SUPERNODE = "logs/supernode"
LOG_DIR_CLIENTAPP = "logs/clientapp"
PROFILE_DIR = "device_profiles"
DEFAULT_PROFILE_NAMESPACE = "default"
COMBINED_PROFILE_POOL_FILE = str(Path(__file__).parent / "traces" / "combined.json")
NETWORK_PROFILE_TRACE_FILE = str(
    Path(__file__).parent / "traces" / "network" / "trace.json"
)
COMPUTE_PROFILE_TRACE_FILE = str(
    Path(__file__).parent / "traces" / "computation" / "trace.json"
)

# Superlink flower
SUPERLINK_CONNECTION_NAME = "local-deployment"

# Ports
SUPERLINK_PORTS = {
    "serverappio": 15001,
    "fleet": 15002,
    "control": 15003,
}  # If you change superlink ports, also update the flwr federation configs. Check `flwr config list`
SUPERNODE_PORT_START = 15100
TOXIPROXY_PROXY_PORT_START = 16100
TOXIPROXY_API_PORT = 8474  # Default ToxiProxy API port

# Note:
# Check the availability of these ports before running, or adjust as needed to avoid conflicts.
# Command to check:
# `ss -tlnp | awk 'NR>1 {print $4}' | grep -oP '(?<=:)\d+' | awk -v lo=15000 -v hi=16200 '$1>=lo && $1<=hi' | sort -n`

AVG_NUM_SAMPLES = 200
ESTIMATED_MODEL_SIZE_KB = 3174.0  # ModernBERT; r=8; target_modules=["Wqkv", "attn.Wo"]

# ── Process Registry ───────────────────────────────────────────────────────────

PID_FILE = ".pids.json" # File to store PIDs of launched processes for cleanup
_launched_procs: list[subprocess.Popen] = []  # Track launched processes within this script's execution

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclasses.dataclass
class SupernodeSpec:
    """Declarative description of one supernode instance."""

    name: str
    node_id: int
    profile: dict
    profile_key: str
    profile_path: str
    clientappio_port: int
    superlink_address: str  # "127.0.0.1:{16100+N}" (via ToxiProxy) or "127.0.0.1:15002"
    total_nodes: int


def get_node_name(node_id: int) -> str:
    return f"supernode-{node_id:03d}"


def get_profile_namespace_dir(namespace: str) -> str:
    namespace_clean = (namespace or DEFAULT_PROFILE_NAMESPACE).strip()
    if not namespace_clean:
        namespace_clean = DEFAULT_PROFILE_NAMESPACE
    return os.path.join(PROFILE_DIR, namespace_clean)


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
    p_up.add_argument("--namespace", type=str, default=DEFAULT_PROFILE_NAMESPACE)
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
    p_prof = sub.add_parser(
        "profiles",
        aliases=["scenario-profiles"],
        help="Sample scenario profiles from traces/combined.json",
    )
    p_prof.add_argument("-n", "--num-clients", type=int, default=100)
    p_prof.add_argument("--combined-path", type=str, default=COMBINED_PROFILE_POOL_FILE)
    p_prof.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Composition skew: 0=balanced, >0=fast-dominant, <0=straggler-dominant",
    )
    p_prof.add_argument("--num-samples", type=int, default=AVG_NUM_SAMPLES)
    p_prof.add_argument("--local-epochs", type=int, default=1)
    p_prof.add_argument("--model-size-kb", type=float, default=ESTIMATED_MODEL_SIZE_KB)
    p_prof.add_argument("--seed", type=int, default=42)
    p_prof.add_argument("--namespace", type=str, default=DEFAULT_PROFILE_NAMESPACE)
    p_prof.add_argument("--show", action="store_true", help="Print summary")

    args = parser.parse_args()

    if args.command == "up":
        logger.debug("Parsed args: %s", args)
        up(
            profile_namespace=args.namespace,
            use_toxiproxy=not args.no_toxiproxy,
        )
    elif args.command == "down":
        down()
    elif args.command == "run":
        run_task(args.overrides)
    elif args.command == "batch":
        run_batch(args.batch_config)
    elif args.command in {"profiles", "scenario-profiles"}:
        profiles = generate_profiles(
            num_clients=args.num_clients,
            combined_path=args.combined_path,
            beta=args.beta,
            num_samples=args.num_samples,
            local_epochs=args.local_epochs,
            model_size_kb=args.model_size_kb,
            seed=args.seed,
            output_dir=get_profile_namespace_dir(args.namespace),
        )
        if args.show:
            _print_profile_summary(profiles)



# ── Up ───────────────────────────────────────────────────────────────────────


def up(
    profile_namespace: str = DEFAULT_PROFILE_NAMESPACE,
    use_toxiproxy: bool = True,
) -> None:
    """Start superlink, configure ToxiProxy, and spawn supernodes."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR_SUPERNODE, exist_ok=True)
    os.makedirs(LOG_DIR_CLIENTAPP, exist_ok=True)

    # Phase 1 — Describe: load pre-generated profiles and build specs
    namespace_dir = get_profile_namespace_dir(profile_namespace)
    logger.info(
        "Loading pre-generated device profiles from namespace '%s' (%s)...",
        profile_namespace,
        namespace_dir,
    )
    profiles = load_profiles(namespace_dir)

    def _superlink_addr(node_id: int) -> str:
        if use_toxiproxy:
            return f"127.0.0.1:{TOXIPROXY_PROXY_PORT_START + node_id}"
        return f"127.0.0.1:{SUPERLINK_PORTS['fleet']}"

    specs = [
        SupernodeSpec(
            name=get_node_name(p["client_id"]),
            node_id=p["client_id"],
            profile=p,
            profile_key=get_node_name(p["client_id"]),
            profile_path=os.path.join(namespace_dir, "all_profiles.json"),
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
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {pool.submit(_launch_supernode, spec): spec for spec in specs}
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                if exc := future.exception():
                    logger.error("Failed to start supernode-%d: %s", spec.node_id, exc)
    except Exception as e:
        logger.error("Error during supernode launch: %s. Initiating teardown.", e)
        down()
        sys.exit(1)

    logger.info("All %d supernodes started.", len(specs))
    logger.info("Active processes: %d", len(_launched_procs))
    logger.info("Saving PIDs of launched processes for cleanup...")
    _save_pids(_launched_procs)
    logger.info("Setup complete. You can now run experiments with 'python setup.py run ...'.")


# ── Superlink ──────────────────────────────────────────────────────────────────


def _build_superlink_cmd() -> list[str]:
    return [
        "flower-superlink",
        "--insecure",
        "--isolation", "subprocess",
        "--serverappio-api-address", f"0.0.0.0:{SUPERLINK_PORTS['serverappio']}",
        "--fleet-api-address",       f"0.0.0.0:{SUPERLINK_PORTS['fleet']}",
        "--control-api-address",     f"0.0.0.0:{SUPERLINK_PORTS['control']}",
    ]


def _launch_superlink(log_dir: str = LOG_DIR) -> subprocess.Popen:
    log_fh = open(Path(log_dir) / "superlink.log", "w")

    # Filter out high-frequency Fleet.PullMessages noise via a grep co-process.
    grep_proc = subprocess.Popen(
        ["grep", "--line-buffered", "-v", "Fleet.PullMessages"],
        stdin=subprocess.PIPE,
        stdout=log_fh,
        stderr=log_fh,
    )

    proc = subprocess.Popen(
        _build_superlink_cmd(),
        stdout=grep_proc.stdin,
        stderr=grep_proc.stdin,
        start_new_session=True,
    )
    grep_proc.stdin.close()

    _launched_procs.append(proc)
    return proc


# ── Supernode ──────────────────────────────────────────────────────────────────


def _build_supernode_cmd(spec: SupernodeSpec) -> list[str]:
    return [
        "flower-supernode",
        "--insecure",
        "--superlink",
        spec.superlink_address,
        "--clientappio-api-address",
        f"0.0.0.0:{spec.clientappio_port}",
        "--isolation",
        "subprocess",
        "--node-config",
        (
            f"partition-id={spec.node_id}"
            f" num-partitions={spec.total_nodes}"
            f" device-profile-path={spec.profile_path}"
            f" device-profile-key={spec.profile_key}"
            f" client-log-path={LOG_DIR_CLIENTAPP}/{spec.name}.log"
        ),
    ]


def _launch_supernode(spec: SupernodeSpec, log_dir: str = LOG_DIR_SUPERNODE) -> subprocess.Popen:

    log_path = Path(log_dir) / f"{spec.name}.log"
    log_fh = open(log_path, "w")

    proc = subprocess.Popen(
        _build_supernode_cmd(spec),
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )

    _launched_procs.append(proc)
    logger.debug(
        "Started %s on port %d via %s",
        spec.name,
        spec.clientappio_port,
        spec.superlink_address,
    )
    return proc


# ── Superlink Health Check ─────────────────────────────────────────────────────


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
    logger.info("Stopping all Flower processes...")

    for pgid in _load_pids():
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    time.sleep(2)

    for pgid in _load_pids():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    _delete_pid_file()
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
        run_task(["task=text_classification", "model=modernbert", "dataset=sst2",
                  "strategy=oort", "num_server_rounds=100"])
    """
    cmd = ["flwr", "run", ".", SUPERLINK_CONNECTION_NAME, "--stream"]

    for ov in overrides:
        cmd.extend(["--run-config", ov])

    logger.info("Running: %s", " ".join(cmd))

    log_path = f"{LOG_DIR}/flower-task.log"
    with open(log_path, "w") as log_file:
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            text=True,
            start_new_session=True,
        )

    logger.info("Experiment started. See log file: %s", log_path)


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
        logger.info("Experiment %d/%d: %s", idx, len(experiments), overrides)
        run_task(overrides)

        time.sleep(5)

    logger.info("Batch complete: %d/%d experiments run.", total, total)


# ── Device Profile Management ─────────────────────────────────────────────────


def _estimate_round_duration_from_profile(
    profile: dict,
    num_samples: int,
    local_epochs: int,
    model_size_kb: float,
) -> float:
    train_time_s = (
        float(profile["computation_latency_ms"]) * num_samples * local_epochs / 1000.0
    )
    comm_time_s = (
        model_size_kb / max(1.0, float(profile["download_kbps"]))
        + model_size_kb / max(1.0, float(profile["upload_kbps"]))
        + float(profile["latency_ms"]) / 1000.0
    )
    return train_time_s + comm_time_s


def sample_by_composition(
    profiles: list[dict],
    n_clients: int,
    beta: float = 0.0,
    num_samples: int = AVG_NUM_SAMPLES,
    local_epochs: int = 1,
    model_size_kb: float = ESTIMATED_MODEL_SIZE_KB,
    seed: int = 42,
) -> list[dict]:
    """Sample n_clients profiles with composition controlled by beta.

    beta=0  -> balanced (uniform by rank)
    beta>0  -> fast-dominant
    beta<0  -> straggler-dominant
    """
    if not profiles:
        raise ValueError("Profile pool is empty; cannot sample scenario profiles.")

    rng = np.random.default_rng(seed)

    durations = np.array(
        [
            _estimate_round_duration_from_profile(
                p,
                num_samples=num_samples,
                local_epochs=local_epochs,
                model_size_kb=model_size_kb,
            )
            for p in profiles
        ],
        dtype=float,
    )
    ranked_idx = np.argsort(durations)
    pool_size = len(profiles)

    ranks = np.arange(1, pool_size + 1, dtype=float)
    weights = ((pool_size - ranks + 1.0) / pool_size) ** beta
    weights = weights / weights.sum()

    chosen_idx = rng.choice(pool_size, size=n_clients, replace=True, p=weights)

    selected_profiles: list[dict] = []
    for new_client_id, ranked_choice_idx in enumerate(chosen_idx):
        original = profiles[int(ranked_idx[int(ranked_choice_idx)])]
        sampled = dict(original)
        sampled["source_pool_index"] = int(ranked_idx[int(ranked_choice_idx)])
        sampled["client_id"] = new_client_id
        if not sampled.get("device_name"):
            sampled.pop("device_name", None)
        selected_profiles.append(sampled)

    return selected_profiles


def _load_combined_profile_pool(combined_path: str) -> tuple[list[dict], float]:
    if not os.path.exists(combined_path):
        raise FileNotFoundError(
            f"Combined profile pool not found: {combined_path}. "
            "Generate it first using traces/combine.ipynb."
        )

    with open(combined_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        pool = data.get("profiles", [])
        metadata = data.get("metadata", {})
        t_min_ms = float(metadata.get("t_min_ms", 0))
    elif isinstance(data, list):
        pool = data
        t_min_ms = 0.0
    else:
        raise ValueError(
            f"Invalid format for combined pool at {combined_path}. "
            "Expected list[dict] or {'profiles': [...], 'metadata': {...}}."
        )

    if not pool:
        raise ValueError(
            f"Combined profile pool at {combined_path} is empty. "
            "Regenerate it using traces/combine.ipynb."
        )

    required_keys = {
        "computation_latency_ms",
        "download_kbps",
        "upload_kbps",
        "latency_ms",
    }
    normalized_pool: list[dict] = []
    for idx, raw in enumerate(pool):
        missing = required_keys.difference(raw.keys())
        if missing:
            raise ValueError(
                f"Profile index {idx} in {combined_path} is missing keys: {sorted(missing)}"
            )

        normalized = {
            "computation_latency_ms": float(raw["computation_latency_ms"]),
            "download_kbps": float(raw["download_kbps"]),
            "upload_kbps": float(raw["upload_kbps"]),
            "latency_ms": float(raw["latency_ms"]),
            "jitter_ms": float(raw.get("jitter_ms", 0.0)),
            "network_type": raw.get("network_type", "unknown"),
        }
        if raw.get("device_name"):
            normalized["device_name"] = raw["device_name"]
        normalized_pool.append(normalized)

    if t_min_ms <= 0:
        t_min_ms = min(p["computation_latency_ms"] for p in normalized_pool)

    return normalized_pool, t_min_ms


def generate_profiles(
    num_clients: int,
    combined_path: str = COMBINED_PROFILE_POOL_FILE,
    beta: float = 0.0,
    num_samples: int = AVG_NUM_SAMPLES,
    local_epochs: int = 1,
    model_size_kb: float = ESTIMATED_MODEL_SIZE_KB,
    seed: int = 42,
    output_dir: str = PROFILE_DIR,
) -> list[dict]:
    """Sample and persist scenario-specific device profiles from combined pool."""
    pool, t_min_ms = _load_combined_profile_pool(combined_path)

    profile_dicts = sample_by_composition(
        profiles=pool,
        n_clients=num_clients,
        beta=beta,
        num_samples=num_samples,
        local_epochs=local_epochs,
        model_size_kb=model_size_kb,
        seed=seed,
    )

    os.makedirs(output_dir, exist_ok=True)

    profiles_by_node = {
        get_node_name(profile["client_id"]): profile for profile in profile_dicts
    }

    all_profiles_path = os.path.join(output_dir, "all_profiles.json")
    with open(all_profiles_path, "w") as f:
        json.dump(profiles_by_node, f, indent=2)
    logger.info("Saved %d profiles to %s", len(profile_dicts), all_profiles_path)

    metadata_path = os.path.join(output_dir, "metadata.json")
    metadata = {
        "t_min_ms": t_min_ms,
        "sampling": {
            "beta": beta,
            "seed": seed,
            "num_clients": num_clients,
            "num_samples": num_samples,
            "local_epochs": local_epochs,
            "model_size_kb": model_size_kb,
            "pool_path": combined_path,
            "pool_size": len(pool),
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        "Saved profile metadata (t_min_ms=%.4f, beta=%.2f) to %s",
        t_min_ms,
        beta,
        metadata_path,
    )

    return profile_dicts


def load_profiles(profile_dir: str = PROFILE_DIR) -> list[dict]:
    """Load previously generated profiles."""
    if not os.path.isdir(profile_dir):
        raise FileNotFoundError(
            f"Profile namespace directory not found: {profile_dir}. "
            "Generate it first with 'python setup.py profiles --namespace=<name>'."
        )

    combined = os.path.join(profile_dir, "all_profiles.json")
    if not os.path.exists(combined):
        raise FileNotFoundError(
            f"No profiles found at {combined}. "
            "Generate them first with 'python setup.py profiles --namespace=<name>'."
        )

    with open(combined) as f:
        profiles = json.load(f)

    return profiles.values()

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

# ── Process Management ─────────────────────────────────────────────────────────

def _save_pids(procs: list[subprocess.Popen]) -> None:
    pids = [os.getpgid(p.pid) for p in procs]
    with open(PID_FILE, "w") as f:
        json.dump(pids, f)
    logger.info("Saved %d PIDs to %s", len(pids), PID_FILE)

def _load_pids() -> list[int]:
    if not os.path.exists(PID_FILE):
        logger.warning("No PID file found at %s — nothing to kill.", PID_FILE)
        return []
    with open(PID_FILE) as f:
        return json.load(f)

def _delete_pid_file() -> None:
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)



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
