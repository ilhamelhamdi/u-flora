#!/bin/bash
#SBATCH --job-name=flowertune-distilbert-lora
#SBATCH --output=results/logs/%j/out.txt
#SBATCH --error=results/logs/%j/err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=1gpu
#SBATCH --partition=dgx-a100
#SBATCH --gpus=1
#SBATCH --mail-user=ilham.abdillah.alhamdi@gmail.com
#SBATCH --mail-type=ALL

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

# Variables
SIF_PATH=/srv/images/python_3.12.9.sif
INSTANCE_NAME=flowertune-distilbert-${DATASET}      
PROJECT_ROOT=$(pwd)
SETUP_ENV_SCRIPT=${PROJECT_ROOT}/scripts/setup-env.sh

export SINGULARITYENV_FLWR_HOME=${PROJECT_ROOT}/.flwr
export SINGULARITYENV_HF_HOME=${PROJECT_ROOT}/.cache/huggingface
export SINGULARITYENV_WANDB_CACHE_DIR=${PROJECT_ROOT}/.cache/wandb

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
        flwr run . --stream --run-config \"dataset='$DATASET' wandb.run-name='dgx-run'\"
    "

echo "Stopping Singularity instance..."
singularity instance stop $INSTANCE_NAME