FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

# Intall dependencies
# Exclude torch since it is already installed in the base image
# Remove the [simulation] extra from flwr (not needed for deployment) 
# --mount=type=cache,target=/root/.cache/uv to speed up subsequent builds by caching the downloaded packages
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i 's/\[simulation\]//' pyproject.toml && \
    sed -i '/"torch==/d' pyproject.toml && \
    uv pip install --system --break-system-packages --no-cache-dir \
    -r pyproject.toml

ENTRYPOINT ["flower-superlink"]