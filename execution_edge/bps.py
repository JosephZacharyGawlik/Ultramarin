"""Implementation shortfall in basis points.

The challenge's evaluation metric. For one one-hour session::

    impl_error = |average_execution_price - close_price| / close_price * 10_000
    penalty    = min(100, volume_to_fill / volume_executed)
    bps        = impl_error * penalty

The per-pair report number is the mean of this BPS value across that pair's
holdout hours. Both the numpy reference simulator and the differentiable
PyTorch variant feed into this function.
"""

from __future__ import annotations

import numpy as np


def compute_bps(
    average_execution_price: float,
    close_price: float,
    volume_to_fill: float,
    volume_executed: float,
) -> float:
    """Return implementation shortfall in basis points for one hour.

    Parameters
    ----------
    average_execution_price : float
        Volume-weighted average price actually achieved by the schedule.
    close_price : float
        Benchmark price (the hour's final close, in our pipeline).
    volume_to_fill : float
        Target volume the schedule was asked to fill.
    volume_executed : float
        Total volume the simulator was able to fill given visible book depth.

    Returns
    -------
    float
        Implementation shortfall in basis points with the volume-fill penalty
        applied. Lower is better. The penalty is capped at 100.
    """
    if volume_executed <= 0 or close_price <= 0 or np.isnan(average_execution_price):
        return float("nan")
    impl_error = abs(average_execution_price - close_price) / close_price * 10_000.0
    penalty = min(100.0, volume_to_fill / volume_executed)
    return float(impl_error * penalty)


def compute_bps_squared(
    average_execution_price,
    close_price,
    volume_to_fill,
    volume_executed,
):
    """BPS-squared variant used as a smooth training loss by the direct-BPS model.

    Parameters and shapes follow ``compute_bps``. Avoids the absolute value so
    that the loss is everywhere differentiable, and squares the relative
    error so that the gradient grows with the magnitude of mispricing.
    """
    impl_error_signed = (average_execution_price - close_price) / close_price * 10_000.0
    penalty = (volume_to_fill / volume_executed).clamp_max(100.0)  # works for torch tensors
    return (impl_error_signed ** 2) * penalty
