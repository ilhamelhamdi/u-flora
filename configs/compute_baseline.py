"""
Per-sample training time (ms) for ModernBERT + LoRA on each GLUE task,
measured on the DGX-A100 testbed.

Populate by running:
    python scripts/profile_compute_baseline.py --task boolq
    python scripts/profile_compute_baseline.py --task mnli

Then paste the reported mean_ms_per_sample values below.
"""

T_ACTUAL_MS_PER_SAMPLE: dict[str, float] = {
    "boolq": 0.0,   # TODO: fill in after profiling
    "mnli": 0.0,   # TODO: fill in after profiling
}
