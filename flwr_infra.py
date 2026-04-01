import subprocess
import time
import os
import sys
import logging
import tomllib
import threading

# (count, cpus) tuples defining client groups.
DEFAULT_CLIENT_GROUPS = [
    (1, 8),
    (1, 4),
    (1, 2),
    (1, 1),
]

# Configuration
SIF_FILE = "flwr.sif"
SUPERLINK_PORTS = {
    "serverappio": 54001,
    "fleet": 54002,
    "control": 54003
}
SUPERNODE_PORT_START = 54100  # Starting port for Supernodes (clientappio API)
CLIENT_GROUPS_CONFIG = "client_groups.toml"

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_client_groups(config_path: str):
    if not os.path.exists(config_path):
        logger.info(
            "Client group config '%s' not found, using defaults",
            config_path,
        )
        return DEFAULT_CLIENT_GROUPS

    try:
        with open(config_path, "rb") as config_file:
            data = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.error("Failed to read '%s': %s", config_path, exc)
        logger.info("Falling back to default client groups")
        return DEFAULT_CLIENT_GROUPS

    raw_groups = data.get("client_groups", [])
    parsed_groups = []

    for index, group in enumerate(raw_groups):
        count = group.get("count")
        cpus = group.get("cpus")

        if not isinstance(count, int) or count < 1:
            logger.error("Invalid count in client_groups[%s]: %r", index, count)
            continue

        if not isinstance(cpus, int) or cpus < 1:
            logger.error("Invalid cpus in client_groups[%s]: %r", index, cpus)
            continue

        parsed_groups.append((count, cpus))

    if not parsed_groups:
        logger.error("No valid client_groups found in '%s'", config_path)
        logger.info("Falling back to default client groups")
        return DEFAULT_CLIENT_GROUPS

    logger.info("Loaded %s client group entries from '%s'", len(parsed_groups), config_path)
    logger.debug("Resolved client groups: %s", parsed_groups)
    return parsed_groups


def up_instances():
    """Spawns the Superlink and the heterogeneous Supernodes."""
    if not os.path.exists(SIF_FILE):
        logger.error("%s not found. Run 'apptainer pull' first.", SIF_FILE)
        return

    # 1. Start Superlink
    logger.info("Launching Superlink (logs: %s/superlink.log)", LOG_DIR)
    logger.debug("Starting instance: superlink")
    subprocess.run(["apptainer", "instance", "start",
                   SIF_FILE, "superlink"], check=False)

    proc = subprocess.Popen([
        "apptainer", "exec", 
        "instance://superlink",
        "flower-superlink", "--insecure", "--isolation", "subprocess",
        "--serverappio-api-address", f"0.0.0.0:{SUPERLINK_PORTS['serverappio']}",
        "--fleet-api-address", f"0.0.0.0:{SUPERLINK_PORTS['fleet']}",
        "--control-api-address", f"0.0.0.0:{SUPERLINK_PORTS['control']}"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)

    # Filter out "Fleet.PullMessages" logs to reduce noise in superlink.log
    def filter_and_write():
        with open(f"{LOG_DIR}/superlink.log", "w") as f:
            for line in iter(proc.stdout.readline, ""):
                if "Fleet.PullMessages" not in line:
                    f.write(line)
                    f.flush()

    threading.Thread(target=filter_and_write, daemon=True).start()
    logger.debug("Superlink process started")

    # Wait for Link to bind ports 9091-9093
    time.sleep(5)

    # 2. Start Heterogeneous Supernodes
    client_groups = load_client_groups(CLIENT_GROUPS_CONFIG)
    node_id = 0
    total_nodes = sum(count for count, _ in client_groups)

    logger.info("Spawning %s heterogeneous supernodes", total_nodes)

    for count, cpus in client_groups:
        for _ in range(count):
            instance_name = f"supernode-{node_id}"
            port = SUPERNODE_PORT_START + node_id

            logger.debug(
                "Node %s config: cpus=%s, port=%s",
                node_id,
                cpus,
                port,
            )

            # Start the background instance shell
            subprocess.run(["apptainer", "instance", "start",
                           SIF_FILE, instance_name], check=False)

            node_log = open(f"{LOG_DIR}/{instance_name}.log", "w")
            exec_cmd = [
                "apptainer", "exec",
                "--env", f"OMP_NUM_THREADS={int(cpus)}",
                "--env", f"MKL_NUM_THREADS={int(cpus)}",
                f"instance://{instance_name}",
                "flower-supernode", "--insecure",
                "--superlink", f"127.0.0.1:{SUPERLINK_PORTS['fleet']}",
                "--clientappio-api-address", f"0.0.0.0:{port}",
                "--isolation", "subprocess",
                "--node-config", f"partition-id={node_id} num-partitions={total_nodes}"
            ]

            subprocess.Popen(exec_cmd, stdout=node_log,
                             stderr=node_log, start_new_session=True)
            logger.info("Started %s on port %s (Virtual CPU(s)): %s)",
                        instance_name, port, cpus)
            node_id += 1

    logger.info("Current Apptainer instance status:")
    subprocess.run(["apptainer", "instance", "list"], check=False)


def down_instances():
    """Stops all running Apptainer instances on this host."""
    logger.info("Stopping all Flower instances")
    # Stopping all is the safest way to ensure no orphaned nodes remain
    result = subprocess.run(
        ["apptainer", "instance", "stop", "--all"], check=False)
    if result.returncode == 0:
        logger.info("Successfully stopped all instances")
    else:
        logger.error("No active instances found or error during shutdown")

def start_flower_task(command: list[str]):
    """Starts the Flower task."""
    file_log_name = f"{LOG_DIR}/flower-task.log"
    task_log = open(file_log_name, "w")
    logger.info("Starting Flower task with command: %s", " ".join(command))
    subprocess.Popen(command, stdout=task_log, stderr=task_log, start_new_session=True)
    logger.info("Flower has been started. Check %s for output.", file_log_name)

if __name__ == "__main__":
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))

    if len(sys.argv) < 2:
        logger.error("Usage: python run_fl.py [up|down]")
        sys.exit(1)

    action = sys.argv[1].lower()
    if action == "up":
        up_instances()
    elif action == "down":
        down_instances()
    elif action == "start":
        start_flower_task(sys.argv[2:])
    else:
        logger.error("Unknown action: %s. Use 'up' or 'down'.", action)
