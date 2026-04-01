#!/bin/bash
echo "Setting up the environment using a container..."

# Use the same image as your services for consistency
IMAGE="flwr/superlink:1.27.0-py3.13-ubuntu24.04"

# Run the setup inside a container
# -v $(pwd):/app mounts your project
# -w /app sets the working directory
docker run --rm \
    -v $(pwd):/app \
    -w /app \
    --entrypoint bash \
    $IMAGE -c "
        pip install uv && \
        if [ -d '.venv' ]; then rm -rf .venv; fi && \
        uv venv .venv && \
        source .venv/bin/activate && \
        sed 's/.*flwr\[simulation\].*//' pyproject.toml | uv pip install -r -
    "

echo "Environment setup complete."

echo "Starting the application using Docker Compose..."

export PROJECT_DIR=$(pwd)
docker compose -f ./docker/deploy-shared/docker-compose.yaml up --build