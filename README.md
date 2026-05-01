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

## Run

```bash
python setup.py up
python setup.py run task=text_classification model=modernbert dataset=boolq
```

## Notes

- If you don’t have CUDA, avoid installing `flash-attn` and use the default attention fallback.
- For non-CUDA runs, set `model.attn-implementation = "auto"` or `"torch"` in `pyproject.toml`.
- On CUDA machines, ensure `nvcc` is available when building `flash-attn`.
