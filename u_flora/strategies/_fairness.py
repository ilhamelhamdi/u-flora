"""ConvergenceFairnessTracker: tracks dM_smooth(t) and computes lambda(t).

lambda(t) in range [0, lambda_max] is a convergence-driven fairness weight.
- High lambda when convergence slows (dM_smooth small vs historical max) -> fairness pressure.
- lambda -> 0 when convergence is fast -> utility dominates selection.
"""

from __future__ import annotations


class ConvergenceFairnessTracker:
    """Tracks smoothed metric improvement dM and computes lambda(t) in range [0, lambda_max]."""

    def __init__(self, lambda_max: float, dm_ema_beta: float) -> None:
        self.lambda_max = float(lambda_max)
        self.dm_ema_beta = float(dm_ema_beta)
        self._prev: float | None = None
        self._prev_prev: float | None = None
        self._dm_smooth: float = 0.0
        self._running_max: float = 0.0

    def update(self, current_metric: float) -> None:
        """Call once per round with the latest eval metric (e.g. accuracy)."""
        if self._prev is not None and self._prev_prev is not None:
            dm = self._prev - self._prev_prev
            self._dm_smooth = (
                self.dm_ema_beta * dm + (1.0 - self.dm_ema_beta) * self._dm_smooth
            )
            self._running_max = max(self._running_max, self._dm_smooth)
        self._prev_prev = self._prev
        self._prev = current_metric

    def lambda_t(self) -> float:
        """Return current lambda(t); 0.0 until enough metric history to compute."""
        if self._running_max <= 0.0:
            return 0.0
        return max(
            0.0,
            min(
                self.lambda_max,
                self.lambda_max * (1.0 - self._dm_smooth / self._running_max),
            ),
        )

    @property
    def dm_smooth(self) -> float:
        return self._dm_smooth

    @property
    def running_max_dm(self) -> float:
        return self._running_max
