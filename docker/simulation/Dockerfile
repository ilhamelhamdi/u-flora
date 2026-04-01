FROM python:3.12.4

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory inside the container
WORKDIR /root

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

# Copy configuration and lock files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy the rest of the application code
COPY . .

# Ensure the project is installed in editable mode inside the container
RUN uv pip install --system --no-cache -e .

# Set the default command (can be overridden by Singularity/Slurm)
CMD ["flwr", "run", "."]