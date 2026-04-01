FROM flwr/supernode:1.27.0-py3.13-ubuntu24.04

WORKDIR /app

# We don't RUN uv pip install here!
# We just set the PATH to where the shared volume will be mounted.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# The entrypoint remains the same
ENTRYPOINT ["flower-supernode"]