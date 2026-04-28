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

# Variables
SIF_PATH=/srv/images/python_3.12.9.sif
INSTANCE_NAME=uflora-testbed      
PROJECT_ROOT=$(pwd)
SETUP_ENV_SCRIPT=${PROJECT_ROOT}/scripts/setup-env.sh

export SINGULARITYENV_FLWR_HOME=${PROJECT_ROOT}/.flwr

# Set environment variables for Singularity to use the local Python user base
export SINGULARITYENV_PYTHONUSERBASE=${PROJECT_ROOT}/.local
export SINGULARITYENV_PATH=$(pwd)/.local/bin:$PATH


echo "Starting Singularity instance..."
singularity instance start --nv \
    --env-file .env \
    --bind ${PROJECT_ROOT}:/root \
    $SIF_PATH \
    $INSTANCE_NAME


singularity exec --nv \
    --cwd /root \
    --env-file .env \
    instance://$INSTANCE_NAME \
    bash -c "
        chmod +x $SETUP_ENV_SCRIPT && $SETUP_ENV_SCRIPT && \

        source .venv/bin/activate && \
        flwr config list && \
        python setup.py batch --batch-config group1-300.yaml --detached\"
    "

# echo "Stopping Singularity instance..."
# singularity instance stop $INSTANCE_NAME

# Wait indefinitely until timeout
while true; do sleep 1; done