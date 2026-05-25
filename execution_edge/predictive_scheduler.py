"""Schedule constructors.

A schedule is a length-60 vector whose entries sum to the per-pair target
volume; the i-th entry is the volume to execute at second i of the final
minute of the trading hour. Positive entries consume ask depth, negative
entries consume bid depth.

This module collects the schedule constructors used in the report:

  TWAP family
      ``twap_uniform``      Uniform across all 60 seconds.
      ``twap_last_k``       Concentrated uniformly in the final K seconds.

  Predictive scheduler (deep-learning pipeline)
      ``ScheduleConfig``                  Hyperparameters for the constructor.
      ``build_schedule_from_forecasts``   Converts a mid-price forecast (and
                                          optional spread / liquidity forecasts)
                                          into a 60-position schedule, with an
                                          optional blend against last-K TWAP.
      ``inverse_distance_softmax``        The "Model" variant used in early DL
                                          experiments. Collapses to TWAP-K
                                          when the forecast carries no signal.

The adaptive scheduling pipeline produces schedules of the same form but
through a different mechanism (binning over hour-level summary features); its
constructor lives in ``execution_edge.candidates`` to keep that pipeline's
moving parts together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------------
# TWAP family
# ----------------------------------------------------------------------------

def twap_uniform(volume_to_fill: float, length: int = 60) -> np.ndarray:
    """Uniform schedule across the entire last minute."""
    out = np.full(length, volume_to_fill / length, dtype=np.float64)
    return out


def twap_last_k(volume_to_fill: float, k: int, length: int = 60) -> np.ndarray:
    """Last-K TWAP: uniform across the final ``k`` seconds, zeros elsewhere.

    The K used in the report is chosen per pair on the dev split alone by
    sweeping K=1..60 and picking the K that minimises mean TWAP-K BPS.
    """
    if not (1 <= k <= length):
        raise ValueError(f"k must be in [1, {length}], got {k}")
    out = np.zeros(length, dtype=np.float64)
    out[length - k:] = volume_to_fill / k
    return out


# ----------------------------------------------------------------------------
# Predictive scheduler (consumes forecasts produced by the DL pipeline)
# ----------------------------------------------------------------------------

@dataclass
class ScheduleConfig:
    """Hyperparameters for ``build_schedule_from_forecasts``.

    The deployed configuration uses ``window=K_pair`` (the dev-selected K
    for the pair) and ``alpha=0.1`` (a small blend weight toward TWAP-K).
    """

    window: int = 14
    alpha: float = 0.1                   # blend weight: model vs TWAP-K
    price_cap: float = 3.0               # cap on price_score multiple vs mean
    liq_min: float = 0.5
    liq_max: float = 2.0
    spread_min: float = 0.5
    spread_max: float = 2.0
    eps: float = 1e-6


def build_schedule_from_forecasts(
    mid_pred: np.ndarray,
    volume_to_fill: float,
    spread_pred: Optional[np.ndarray] = None,
    liq_pred: Optional[np.ndarray] = None,
    cfg: ScheduleConfig = ScheduleConfig(),
) -> np.ndarray:
    """Convert per-second forecasts into a 60-position schedule.

    The score per second combines a directional price favourability term
    (small distance from the predicted close gets a higher score), an
    optional liquidity tilt (deeper top of book gets a higher score), and an
    optional spread tilt (tighter spread gets a higher score). Scores are
    masked to the final ``cfg.window`` seconds, normalised, multiplied by the
    target volume, and blended with last-K TWAP at weight ``cfg.alpha``. The
    final vector is renormalised so it sums exactly to ``volume_to_fill``.

    Parameters
    ----------
    mid_pred : ndarray, shape ``(60,)``
        Per-second mid-price forecast for the last minute.
    volume_to_fill : float
        Per-pair target volume.
    spread_pred, liq_pred : ndarray, shape ``(60,)``, optional
        Spread and top-of-book liquidity forecasts. The deployed pipeline
        does not pass these (mid-price predictions only).
    cfg : ScheduleConfig
        Hyperparameters; defaults match the deployed configuration.

    Returns
    -------
    ndarray, shape ``(60,)``
        Schedule summing to ``volume_to_fill``.
    """
    assert mid_pred.shape[0] == 60, "Need a 60-second forecast."
    p_close = mid_pred[-1]

    price_score = 1.0 / (np.abs(mid_pred - p_close) + cfg.eps)
    price_score = np.minimum(price_score, cfg.price_cap * price_score.mean())

    liq_score = np.ones_like(price_score)
    if liq_pred is not None:
        mean_liq = np.mean(liq_pred) + cfg.eps
        liq_score = np.clip(liq_pred / mean_liq, cfg.liq_min, cfg.liq_max)

    spread_score = np.ones_like(price_score)
    if spread_pred is not None:
        mean_spread = np.mean(spread_pred) + cfg.eps
        spread_score = np.clip(mean_spread / (spread_pred + cfg.eps), cfg.spread_min, cfg.spread_max)

    score = price_score * liq_score * spread_score

    # Restrict to last-cfg.window seconds.
    start = 60 - cfg.window
    mask = np.zeros_like(score)
    mask[start:] = 1.0
    score = score * mask

    if score.sum() <= 0:
        # Degenerate signal: fall back to last-K TWAP.
        return twap_last_k(volume_to_fill, cfg.window).astype(np.float32)

    weights = score / score.sum()
    model_sched = weights * volume_to_fill

    twap = np.zeros(60, dtype=np.float64)
    twap[start:] = volume_to_fill / cfg.window
    blended = cfg.alpha * model_sched + (1.0 - cfg.alpha) * twap
    blended = blended * (volume_to_fill / blended.sum())
    return blended.astype(np.float32)


def inverse_distance_softmax(
    mid_pred: np.ndarray,
    volume_to_fill: float,
    k: int,
) -> np.ndarray:
    """Inverse-distance softmax schedule (the ``Model`` variant).

    Within the last K seconds, weights are ``w_i = (1 - d_i / sum(d)) / Z``
    where ``d_i = |p_close_hat - p_i_hat|`` is the predicted distance to the
    predicted close. The schedule collapses to last-K TWAP when all distances
    are equal (i.e. when the model carries no directional signal). Outside
    the last K seconds the schedule is zero.

    This variant is included for completeness because it was the first
    scheduler used in the project; the deployed scheduler is the predictive
    scheduler above, which blends with TWAP-K at ``alpha=0.1``.
    """
    if not (1 <= k <= 60):
        raise ValueError(f"k must be in [1, 60], got {k}")

    out = np.zeros(60, dtype=np.float64)
    pred_close = mid_pred[-1]
    window = mid_pred[-k:]
    d = np.abs(pred_close - window)
    s = d.sum()
    if s > 0:
        w = 1 - (d / s)
        w = w / w.sum()
    else:
        w = np.full(k, 1.0 / k)
    out[-k:] = w * volume_to_fill
    return out
