#!/bin/bash
# setup-env.sh

echo "Starting setup-env.sh"
echo "Setting up the environment..."

echo "Installing uv..."

pip install --user uv --break-system-packages

if [ -d ".venv" ]; then
    rm -rf .venv
fi

echo "Creating virtual environment..."
uv venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install -r pyproject.toml --break-system-packages
uv pip install -e .

echo "Environment setup complete."
echo "Completed setup-env.sh"