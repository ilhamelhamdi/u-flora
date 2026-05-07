#!/bin/bash
#SBATCH --job-name=UFLoRA
#SBATCH --output=results/logs/%j/out.txt
#SBATCH --error=results/logs/%j/err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --qos=1gpu
#SBATCH --partition=dgx-a100
#SBATCH --gpus=1
#SBATCH --mail-user=ilham.abdillah.alhamdi@gmail.com
#SBATCH --mail-type=ALL


# 1. Load Host Modules (If your cluster uses 'module load')
# module load cuda/12.x  # Only if needed for runtime libraries
PROJECT_ROOT=$(pwd)
WHEELS_PATH=${HOME}/wheels

# 2. Ensure uv is available
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 3. Environment Setup
echo "Syncing environment..."
# uv python install 3.12 # Ensures the correct python version is used
uv venv .venv --python 3.12
source .venv/bin/activate

# 4. Install pre-built flash-attn (Bypasses lack of nvcc)
echo "Installing pre-built wheels..."
uv pip install "${WHEELS_PATH}/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"

# 5. Install project dependencies
uv pip install -e .


if [ -f .env ]; then
    echo "Loading environment variables..."
    set -a            # Automatically export all variables
    source .env
    set +a            # Turn off automatic exporting
fi

# 6. Run the workload
echo "Starting Flower Simulation..."
flwr config list
python setup.py batch --batch-config oort-tifl-mnli.yaml --detach

# Wait indefinitely until timeout
while true; do sleep 1; done