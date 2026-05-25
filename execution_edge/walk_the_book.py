"""Walk-the-book execution simulator.

This module exposes two implementations of the same simulator.

``simulate_walk_the_book`` is the reference numpy implementation provided by
the challenge organisers and shipped under ``data/`` in this repository. Every
BPS number reported in the paper is produced by this implementation.

``differentiable_walk_the_book`` is a PyTorch re-implementation of the same
walking logic. It supports autograd and is used by the direct-BPS experiment
in ``notebooks/04_direct_bps.ipynb`` to backpropagate gradients of execution
cost into the schedule. It has been verified to agree with the reference
implementation to floating-point precision on the project's holdout data.

Both functions take five inputs: a 60-position schedule (one number per
second of the last minute), and four per-second arrays of ask prices, ask
volumes, bid prices, and bid volumes, each of shape ``[60, 5]`` covering
the top five levels of the book.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Re-export the reference simulator so callers can use a single namespace.
import sys

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

try:
    from simulate_walk_the_book import simulate_walk_the_book as _reference_simulator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import simulate_walk_the_book from data/. Ensure data/simulate_walk_the_book.py is in place."
    ) from exc

simulate_walk_the_book = _reference_simulator


def differentiable_walk_the_book(
    positions,
    ask_prices,
    ask_volumes,
    bid_prices,
    bid_volumes,
    eps: float = 1e-12,
):
    """PyTorch reimplementation of the reference simulator.

    Same semantics as ``simulate_walk_the_book`` but with autograd support and
    no Python loops over book levels: vectorised over the level dimension so
    gradients flow back through the schedule. Positive entries in
    ``positions`` consume ask depth (a buy); negative entries consume bid
    depth (a sell). Mixed-sign schedules are supported.

    Parameters
    ----------
    positions : torch.Tensor
        Shape ``[60]``. Signed volume to execute at each second of the last
        minute.
    ask_prices, ask_volumes, bid_prices, bid_volumes : torch.Tensor
        Shape ``[60, 5]`` each. The first axis is seconds (0 to 59) and the
        second axis is book level (1 to 5).
    eps : float
        Small constant to avoid division-by-zero in the VWAP computation.

    Returns
    -------
    total_volume_executed : torch.Tensor
        Scalar; the absolute volume actually filled.
    average_price : torch.Tensor
        Scalar; the volume-weighted average price across all filled volume.
    """
    import torch  # local import keeps numpy-only callers light

    is_buy = positions >= 0
    abs_positions = positions.abs()

    # Use ask side for buy seconds, bid side for sell seconds.
    prices = torch.where(is_buy.unsqueeze(-1), ask_prices, bid_prices)
    depths = torch.where(is_buy.unsqueeze(-1), ask_volumes, bid_volumes)

    # NaN-safe (treat missing levels as having zero depth at price NaN; we
    # mask the contribution below).
    valid_level = ~(torch.isnan(prices) | torch.isnan(depths))
    safe_depths = torch.where(valid_level, depths, torch.zeros_like(depths))
    safe_prices = torch.where(valid_level, prices, torch.zeros_like(prices))

    # Walk levels left to right, tracking remaining demand. The "consumed at
    # each level" is min(remaining, depth) and the remaining decrements.
    remaining = abs_positions.clone()
    consumed_list: list[torch.Tensor] = []
    for level in range(safe_depths.shape[-1]):
        depth_l = safe_depths[..., level]
        consumed_l = torch.minimum(remaining, depth_l)
        consumed_list.append(consumed_l)
        remaining = remaining - consumed_l
    consumed = torch.stack(consumed_list, dim=-1)  # [60, 5]

    notional = (consumed * safe_prices).sum(dim=-1)         # [60]
    filled = consumed.sum(dim=-1)                            # [60]

    total_filled = filled.sum()
    total_notional = notional.sum()
    vwap = total_notional / (total_filled + eps)
    return total_filled, vwap
