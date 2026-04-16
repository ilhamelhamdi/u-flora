"""
Per-sample training time (ms) for ModernBERT + LoRA on each GLUE task,
measured on the DGX-A100 testbed.

Populate by running:
    python scripts/profile_compute_baseline.py --task sst2
    python scripts/profile_compute_baseline.py --task qnli
    python scripts/profile_compute_baseline.py --task qqp

Then paste the reported mean_ms_per_sample values below.
"""

T_ACTUAL_MS_PER_SAMPLE: dict[str, float] = {
    "sst2": 0.0,   # TODO: fill in after profiling
    "qnli": 0.0,   # TODO: fill in after profiling
    "qqp":  0.0,   # TODO: fill in after profiling
}
