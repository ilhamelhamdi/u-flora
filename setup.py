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
# FedScale device-capacity trace: { "<client_id>": {"computation": ms/sample,
#   "communication": kbps}, ... }
FEDSCALE_PROFILE_POOL_FILE = str(
    Path(__file__).parent / "traces" / "client_device_capacity.json"
)
# FedScale behavior trace (active/inactive timeline per client):
#   { "<client_id>": {"duration", "finish_time", "active": [...],
#                     "inactive": [...], "model"}, ... }
FEDSCALE_BEHAVIOR_TRACE_FILE = str(
    Path(__file__).parent / "traces" / "client_behave_trace.json"
)

# Superlink flower
SUPERLINK_CONNECTION_NAME = "local"

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

PID_FILE = ".pids.json"  # File to store PIDs of launched processes for cleanup
_launched_procs: list[subprocess.Popen] = (
    []
)  # Track launched processes within this script's execution

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
    p_batch.add_argument(
        "--detach",
        action="store_true",
        help="Run batch in background and exit immediately",
    )
    p_batch.add_argument(
        "--reset",
        action="store_true",
        help="Ignore any saved state and re-run all experiments from scratch",
    )

    # --- profiles ---
    p_prof = sub.add_parser(
        "profiles",
        aliases=["scenario-profiles"],
        help="Sample scenario profiles from the FedScale trace.",
    )
    p_prof.add_argument("-n", "--num-clients", type=int, default=100)
    p_prof.add_argument("--fedscale-path", type=str, default=FEDSCALE_PROFILE_POOL_FILE)
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
        if args.reset:
            state_path = _batch_state_path(args.batch_config)
            if state_path.exists():
                state_path.unlink()
                logger.info("Cleared batch state: %s", state_path)
        if args.detach:
            _detach_batch(args.batch_config)
        else:
            run_batch(args.batch_config)
    elif args.command in {"profiles", "scenario-profiles"}:
        profiles = generate_profiles(
            num_clients=args.num_clients,
            fedscale_path=args.fedscale_path,
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
            profile_path=os.path.join(namespace_dir, "profiles.json"),
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
    logger.info(
        "Setup complete. You can now run experiments with 'python setup.py run ...'."
    )


# ── Superlink ──────────────────────────────────────────────────────────────────


def _build_superlink_cmd() -> list[str]:
    return [
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


def _launch_supernode(
    spec: SupernodeSpec, log_dir: str = LOG_DIR_SUPERNODE
) -> subprocess.Popen:

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


def run_task(
    overrides: list[str],
    block: bool = False,
    name: str | None = None,
    superlink_connection: str = SUPERLINK_CONNECTION_NAME,
    max_duration_s: float | None = None,
) -> int | None:
    """Run a single FL experiment via ``flwr run``.

    Args:
        max_duration_s: Hard wall-time limit for the experiment. If the
            ``flwr run`` process is still alive after this many seconds it
            is killed with SIGKILL and the return code is set to -1.
            ``None`` means no limit (original behaviour).

    Example:
        run_task(["task=text_classification", "model=modernbert", "dataset=sst2",
                  "strategy=oort", "num_server_rounds=100"])
    """
    cmd = ["flwr", "run", ".", superlink_connection, "--stream"]

    def _format_override_value(raw_value: str) -> str:
        value = raw_value.strip()
        if not value:
            return '""'

        lower = value.lower()
        if (
            lower in {"true", "false", "null", "none"}
            or value.replace(".", "", 1).lstrip("-").isdigit()
        ):
            return value

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value

        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'

    parsed_overrides = []
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: {override!r}. Expected <key>=<value>."
            )
        key, raw_value = override.split("=", 1)
        parsed_overrides.append(f"{key}={_format_override_value(raw_value)}")

    formatted_overrides = " ".join(parsed_overrides)
    if formatted_overrides:
        cmd.extend(["--run-config", formatted_overrides])

    logger.info("Running: %s", " ".join(cmd))

    if name:
        log_dir = f"{LOG_DIR}/{name}"
        os.makedirs(log_dir, exist_ok=True)
        log_path = f"{log_dir}/flower-task.log"
    else:
        log_path = f"{LOG_DIR}/flower-task.log"

    with open(log_path, "w") as log_file:
        if block:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                text=True,
            )
            try:
                proc.wait(timeout=max_duration_s)
            except subprocess.TimeoutExpired:
                logger.error(
                    "Experiment %s exceeded max duration of %.0fs — killing.",
                    name or "unknown",
                    max_duration_s,
                )
                proc.kill()
                proc.wait()
                return -1
            rc = proc.returncode
            if rc != 0:
                logger.error(
                    "Experiment failed (exit %d). See log file: %s",
                    rc,
                    log_path,
                )
            return rc

        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            text=True,
            start_new_session=True,
        )

    logger.info("Experiment started. See log file: %s", log_path)
    return None


def _batch_state_path(batch_config_path: str) -> Path:
    """Return path to the resume-state file for a given batch config."""
    stem = Path(batch_config_path).stem
    return Path(LOG_DIR) / f"batch-state-{stem}.json"


def _load_batch_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def _save_batch_state(state_path: Path, state: dict) -> None:
    os.makedirs(state_path.parent, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def run_batch(batch_config_path: str) -> None:
    """Run a batch of experiments sequentially.

    The batch config is a YAML file under ``u_flora/config/groups`` and
    lists experiments with a base config name (from ``u_flora/config/defaults``)
    plus extra overrides:

    ```yaml
    experiments:
      - name: "random_sst2"
        base: random_sst2
        overrides:
          - "num_server_rounds=100"
      - name: "oort_sst2"
        base: oort_sst2
        overrides:
          - "num_server_rounds=100"
    ```

    Resume behaviour: a state file is written to ``logs/batch-state-<name>.json``
    after each experiment completes.  If the batch process is restarted, already-
    completed experiments are skipped automatically.  Delete the state file to
    force a full re-run.
    """
    import yaml

    groups_dir = Path(__file__).parent / "u_flora" / "config" / "groups"
    defaults_dir = Path(__file__).parent / "u_flora" / "config" / "defaults"

    config_path = Path(batch_config_path)
    if not config_path.exists():
        config_path = groups_dir / batch_config_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"Batch config not found: {batch_config_path} (or {config_path})"
        )

    with open(config_path) as f:
        batch = yaml.safe_load(f)

    superlink_connection_name = batch.get(
        "superlink-connection", SUPERLINK_CONNECTION_NAME
    )
    # Hard wall-time limit per experiment (seconds). Kills a hung flwr run process.
    # Default: 4 hours. Override in the batch YAML with: max-experiment-duration-s: 7200
    max_experiment_duration_s: float | None = batch.get(
        "max-experiment-duration-s", 4 * 3600
    )

    experiments = batch.get("experiments", [])
    total = len(experiments)

    state_path = _batch_state_path(batch_config_path)
    state = _load_batch_state(state_path)
    completed_set = set(state["completed"])

    if completed_set:
        logger.info(
            "Resuming batch — %d/%d experiments already completed: %s",
            len(completed_set),
            total,
            sorted(completed_set),
        )

    skipped = 0
    for idx, exp in enumerate(experiments, 1):
        name = exp.get("name", f"experiment_{idx}")
        base = exp.get("base")
        overrides = exp.get("overrides", [])

        if name in completed_set:
            logger.info("Experiment %d/%d (%s): already completed — skipping.", idx, total, name)
            skipped += 1
            continue

        base_overrides: list[str] = []
        if base:
            base_path = defaults_dir / f"{base}.yaml"
            if not base_path.exists():
                raise FileNotFoundError(f"Base config not found: {base} ({base_path})")
            with open(base_path) as base_f:
                base_cfg = yaml.safe_load(base_f) or {}
            base_overrides = base_cfg.get("overrides", [])

        merged_overrides = [*base_overrides, *overrides]
        logger.info(
            "Experiment %d/%d (%s, base=%s): %s",
            idx,
            total,
            name,
            base or "none",
            merged_overrides,
        )
        rc = run_task(
            merged_overrides,
            block=True,
            name=name,
            superlink_connection=superlink_connection_name,
            max_duration_s=max_experiment_duration_s,
        )
        if rc not in (0, None):
            logger.error("Experiment %s failed (exit %d)", name, rc)
            state["failed"].append({"name": name, "exit_code": rc})
        else:
            state["completed"].append(name)
            completed_set.add(name)

        _save_batch_state(state_path, state)
        time.sleep(5)

    done = len(completed_set) - skipped
    logger.info(
        "Batch complete: %d newly run, %d skipped (already done), %d total. State: %s",
        done,
        skipped,
        total,
        state_path,
    )


def _detach_batch(batch_config_path: str) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "batch",
        "--batch-config",
        batch_config_path,
    ]
    log_path = f"{LOG_DIR}/batch-detached.log"
    proc = _spawn_detached_process(cmd, log_path)
    logger.info(
        "Detached batch started (pid %d). See log file: %s",
        proc.pid,
        log_path,
    )


def _spawn_detached_process(cmd: list[str], log_path: str) -> subprocess.Popen:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = open(log_path, "a")
    cwd = str(Path(__file__).parent)

    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,
        cwd=cwd,
        start_new_session=True,
    )


# ── Device Profile Management ─────────────────────────────────────────────────


def _estimate_round_duration_from_profile(
    profile: dict,
    num_samples: int,
    local_epochs: int,
    model_size_kb: float,
) -> float:
    """Composition-ranking helper.

    NOTE: this uses the RAW FedScale ``computation`` value (ms/sample on the
    benchmark) directly without anchoring to a per-task ``t_actual_ms``.
    That is fine here because we only need to *rank* profiles by relative
    speed; the absolute number of seconds is not used downstream of sampling.
    """
    raw_comp_ms = float(profile.get("computation", 0.0))
    comm_kbps = float(profile.get("communication", 0.0))

    compute_s = raw_comp_ms * num_samples * local_epochs / 1000.0
    # 8 converts kilobytes to kilobits to match `comm_kbps`; matches utils/timing.py.
    comm_s = (
        2.0 * 8.0 * model_size_kb / comm_kbps
        if comm_kbps > 0 and model_size_kb > 0
        else 0.0
    )
    return compute_s + comm_s


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
        # Deep-copy behavior so per-client time_offset_s doesn't leak across clients
        # that happened to draw the same pool entry.
        if "behavior" in sampled:
            beh = dict(sampled["behavior"])
            beh["time_offset_s"] = _sample_online_offset(beh, rng)
            sampled["behavior"] = beh
        sampled["source_pool_index"] = int(ranked_idx[int(ranked_choice_idx)])
        sampled["client_id"] = new_client_id
        selected_profiles.append(sampled)

    return selected_profiles


def _sample_online_offset(behavior: dict, rng: np.random.Generator) -> float:
    """Sample ``time_offset_s`` uniformly from the client's online intervals.

    Guarantees the client is online at virtual_time=0 (so the first FL round
    starts with every client available — important for TiFL pretraining and
    for stable behavior in the warmup phase). As virtual time progresses,
    the marginal online distribution converges to the population's natural
    ratio.
    """
    active = behavior.get("active", [])
    inactive = behavior.get("inactive", [])
    finish_time = float(behavior.get("finish_time", 0.0))
    if not active or not inactive or finish_time <= 0:
        return 0.0

    intervals = [(float(a), float(b)) for a, b in zip(active, inactive) if b > a]
    total_online = sum(b - a for a, b in intervals)
    if total_online <= 0:
        return 0.0

    r = float(rng.uniform(0.0, total_online))
    acc = 0.0
    for a, b in intervals:
        width = b - a
        if r < acc + width:
            return a + (r - acc)
        acc += width
    # Fallback (numerical edge): return the very last online instant.
    return intervals[-1][1] - 1e-6


def _load_fedscale_profile_pool(
    fedscale_path: str,
    behavior_path: str = FEDSCALE_BEHAVIOR_TRACE_FILE,
) -> tuple[list[dict], float]:
    """Load the FedScale capacity + behavior traces and pair them.

    The capacity trace has shape::
        { "<client_id>": {"computation": ms/sample, "communication": kbps}, ...}
    The behavior trace has shape::
        { "<client_id>": {"duration", "finish_time",
                          "active": [...], "inactive": [...], "model"}, ...}

    Only clients present in BOTH traces are kept; pairing preserves the joint
    distribution from the FedScale dataset.

    Returns (pool, t_min_ms) where each pool item is::

        {
            "computation":   float,
            "communication": float,
            "behavior": {
                "active":      [...],
                "inactive":    [...],
                "finish_time": int,
                "model":       str,
            },
            "trace_id": str,
        }
    """
    if not os.path.exists(fedscale_path):
        raise FileNotFoundError(
            f"FedScale capacity trace not found: {fedscale_path}. "
            "Download from the FedScale repo into traces/."
        )
    if not os.path.exists(behavior_path):
        raise FileNotFoundError(
            f"FedScale behavior trace not found: {behavior_path}. "
            "Download from the FedScale repo into traces/."
        )

    with open(fedscale_path) as f:
        capacity = json.load(f)
    with open(behavior_path) as f:
        behavior = json.load(f)

    if not isinstance(capacity, dict) or not isinstance(behavior, dict):
        raise ValueError("Invalid FedScale traces; expected dicts keyed by client_id.")

    pool: list[dict] = []
    for raw_id, cap_entry in capacity.items():
        beh_entry = behavior.get(raw_id)
        if not isinstance(cap_entry, dict) or not isinstance(beh_entry, dict):
            continue
        comp = cap_entry.get("computation")
        comm = cap_entry.get("communication")
        active = beh_entry.get("active")
        inactive = beh_entry.get("inactive")
        finish = beh_entry.get("finish_time")
        if comp is None or comm is None:
            continue
        if not active or not inactive or not finish:
            continue
        pool.append(
            {
                "computation": float(comp),
                "communication": float(comm),
                "behavior": {
                    "active": list(active),
                    "inactive": list(inactive),
                    "finish_time": int(finish),
                    "model": beh_entry.get("model", "unknown"),
                },
                "trace_id": str(raw_id),
            }
        )

    if not pool:
        raise ValueError(
            f"Empty intersection of capacity ({fedscale_path}) and behavior "
            f"({behavior_path}) traces."
        )

    logger.info(
        "Loaded FedScale pool: %d clients (intersection of capacity=%d × behavior=%d)",
        len(pool),
        len(capacity),
        len(behavior),
    )

    t_min_ms = min(p["computation"] for p in pool if p["computation"] > 0)
    return pool, t_min_ms


def generate_profiles(
    num_clients: int,
    fedscale_path: str = FEDSCALE_PROFILE_POOL_FILE,
    beta: float = 0.0,
    num_samples: int = AVG_NUM_SAMPLES,
    local_epochs: int = 1,
    model_size_kb: float = ESTIMATED_MODEL_SIZE_KB,
    seed: int = 42,
    output_dir: str = PROFILE_DIR,
) -> list[dict]:
    """Sample and persist scenario-specific device profiles from the FedScale pool."""
    pool, t_min_ms = _load_fedscale_profile_pool(fedscale_path)

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

    profiles_path = os.path.join(output_dir, "profiles.json")
    with open(profiles_path, "w") as f:
        json.dump(profiles_by_node, f, indent=2)
    logger.info("Saved %d profiles to %s", len(profile_dicts), profiles_path)

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
            "pool_path": fedscale_path,
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

    combined = os.path.join(profile_dir, "profiles.json")
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

    # FedScale shape: single symmetric `communication` kbps — apply to both streams.
    rate_kb_per_s = max(1, int(float(profile["communication"]) / 8))
    toxics = [
        (f"bw_down_{cid}", "bandwidth", "downstream", {"rate": rate_kb_per_s}),
        (f"bw_up_{cid}", "bandwidth", "upstream", {"rate": rate_kb_per_s}),
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
    """Print a concise summary of the generated FedScale-shaped profiles."""
    n = len(profiles)
    comp = sorted(p["computation"] for p in profiles)
    bw = sorted(p["communication"] for p in profiles)

    online_frac: list[float] = []
    for p in profiles:
        beh = p.get("behavior")
        if beh and beh.get("finish_time", 0) > 0:
            # duration field isn't carried through; recompute from active/inactive
            active = beh.get("active", [])
            inactive = beh.get("inactive", [])
            online = sum(max(0, b - a) for a, b in zip(active, inactive))
            online_frac.append(online / float(beh["finish_time"]))
    online_frac.sort()

    def percentile(vals: list, pct: int) -> float:
        return vals[min(int(len(vals) * pct / 100), len(vals) - 1)]

    print(f"\n{'=' * 70}")
    print(f"  {n} Device Profiles (FedScale)")
    print(f"{'=' * 70}")
    print(
        f"  {'Metric':<25} {'P10':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P90':>10}"
    )
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    rows = [
        ("Compute (ms/sample)", comp, "{:.1f}"),
        ("Bandwidth (kbps)", bw, "{:.1f}"),
    ]
    if online_frac:
        rows.append(("Online fraction", online_frac, "{:.3f}"))

    for label, vals, fmt in rows:
        row = [fmt.format(percentile(vals, p)) for p in [10, 25, 50, 75, 90]]
        print(f"  {label:<25} {'':>3}".rstrip() + "  ".join(f"{v:>10}" for v in row))

    print(f"\n  Heterogeneity (compute): {comp[-1] / max(comp[0], 0.01):.1f}x")
    print(f"  Heterogeneity (bandwidth): {bw[-1] / max(bw[0], 0.01):.1f}x")
    if online_frac:
        print(
            f"  Online-fraction range: {online_frac[0]:.2f}–{online_frac[-1]:.2f} "
            f"(median {online_frac[len(online_frac)//2]:.2f})"
        )


if __name__ == "__main__":
    main()
