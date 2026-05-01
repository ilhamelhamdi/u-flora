# U-Flora

U-Flora is an experiment orchestrator for federated LoRA fine-tuning with Flower, with support for heterogeneous device and network profiles.

## Quick start (Windows, no CUDA)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
uv sync
```

## CUDA setup (Linux VM)

`flash-attn` is optional and only needed on CUDA-capable machines. Install it via the `cuda` extra.

```bash
python -m venv .venv
source .venv/bin/activate

# Install CUDA-enabled PyTorch (recommended: cu124)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install project + CUDA extra (pip)
pip install ".[cuda]" --no-build-isolation
```

If you prefer `uv`, run:

```bash
uv pip install hatchling
uv pip install "[cuda]" --no-build-isolation
```

### Installing a prebuilt `flash-attn` wheel (no `nvcc` on target VM)

If your **run VM** has no `nvcc`, build the wheel on a **build VM** that has CUDA
toolkit installed, then transfer the wheel.

On the build VM (with `nvcc`):

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
PY

rm -rf /tmp/wheels
mkdir -p /tmp/wheels

# Force a fresh build using the current torch (no deps, no cache)
pip wheel flash-attn==2.8.3 -w /tmp/wheels --no-build-isolation --no-deps --no-cache-dir --no-binary=:all:
```

Transfer to the run VM:

```bash
scp /tmp/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl username@host:~/wheels/
```

Install on the run VM (no `nvcc` required):

```bash
source .venv/bin/activate
uv pip install ~/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
```

## Run

```bash
python setup.py up
python setup.py run task=text_classification model=modernbert dataset=boolq
```

## Notes

- If you don’t have CUDA, avoid installing `flash-attn` and use the default attention fallback.
- For non-CUDA runs, set `model.attn-implementation = "auto"` or `"torch"` in `pyproject.toml`.
- On CUDA machines, ensure `nvcc` is available when building `flash-attn`.
