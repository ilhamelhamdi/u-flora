"""FedScale-trace-driven client availability oracle.

A behavior dict has the shape::

    {
        "active":   [t1, t2, ...],   # sorted timestamps when client comes online
        "inactive": [t1, t2, ...],   # sorted timestamps when client goes offline
        "finish_time":   <int>,      # trace span; trace wraps after this
        "time_offset_s": <float>,    # per-client phase offset (avoid synchronized starts)
    }

At any virtual time ``t``, the client is online iff::

    t_mod = (t + offset) % finish_time
    bisect_right(active, t_mod) > bisect_right(inactive, t_mod)
"""

from __future__ import annotations

from bisect import bisect_right


def is_available(behavior: dict, virtual_time_s: float) -> bool:
    """Return True iff the client is online at the given virtual time."""
    finish_time = float(behavior.get("finish_time", 0.0))
    if finish_time <= 0:
        # Missing or zero-span trace → treat as always-online (benign default).
        return True

    offset = float(behavior.get("time_offset_s", 0.0))
    t_mod = (float(virtual_time_s) + offset) % finish_time

    active = behavior.get("active", [])
    inactive = behavior.get("inactive", [])
    return bisect_right(active, t_mod) > bisect_right(inactive, t_mod)
