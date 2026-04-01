#!/bin/bash
#SBATCH --job-name=flowertune-distilbert-lora
#SBATCH --output=results/logs/%j/out.txt
#SBATCH --error=results/logs/%j/err.txt
#SBATCH --ntasks=1
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
IMAGE_NAME=ilhamelhamdi/flowertune-distilbert-lora:latest
PROJECT_ROOT=$(pwd)
INSTANCE_NAME=flowertune-distilbert-${DATASET}          


singularity exec --nv \
    --bind ${PROJECT_ROOT}:/root \
    --env-file .env \
    --pwd /root \
    docker://$IMAGE_NAME \
    flwr run . \
        --stream \
        --run-config "dataset='$DATASET'" \
        --run-config "wandb.run-name='dgx-run'"