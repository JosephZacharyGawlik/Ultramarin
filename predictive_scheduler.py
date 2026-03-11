#!/usr/bin/env python3
"""
Leakage-safe multi-horizon forecaster + scheduler for last-minute execution.

Design goals:
- Use only the first 59 minutes (0:00–58:59) as input.
- Predict the full last-minute path (59:00–59:59) for mid/spread/liquidity.
- Turn forecasts into a 60-length position vector with no last-minute look-ahead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Keep OpenMP usage sandbox-friendly.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_CREATE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# Ensure local .venv packages are discoverable if present.
REPO_ROOT = Path(__file__).resolve().parent
VENV_SITE = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if str(VENV_SITE) not in sys.path and VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))


# -----------------------------
# Data
# -----------------------------

DEFAULT_FEATURES: List[str] = [
    "midprice",
    "spread",
    "spread_pct",
    "ask_vol_1",
    "bid_vol_1",
    "ask_vol_2",
    "bid_vol_2",
    "ask_vol_3",
    "bid_vol_3",
    "ask_vol_4",
    "bid_vol_4",
    "ask_vol_5",
    "bid_vol_5",
    "total_depth",
    "imbalance",
    "volume",
    "mid_return_1s",
    "volume_roll_mean_5",
    "spread_roll_mean_5",
    "imbalance_roll_mean_5",
]

TARGET_HEADS: List[str] = ["midprice", "spread", "top_liquidity"]


def _to_float32(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return df.loc[:, cols].astype(np.float32).to_numpy()


class LastMinuteDataset(Dataset):
    """
    Builds samples of (59-minute history, 60-second targets) per anonymized_id.
    Expects:
      x_df: first 59 minutes (matching X_train schema + engineered features).
      y_df: last minute (matching y_train schema + same engineered features).
    """

    def __init__(
        self,
        x_df: pd.DataFrame,
        y_df: pd.DataFrame,
        feature_cols: Sequence[str] = DEFAULT_FEATURES,
        target_heads: Sequence[str] = TARGET_HEADS,
        input_window: int = 59 * 60,
    ):
        self.feature_cols = list(feature_cols)
        self.target_heads = list(target_heads)
        self.input_window = input_window
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for hour_id, x_hour in x_df.groupby("anonymized_id"):
            y_hour = y_df[y_df["anonymized_id"] == hour_id]
            if len(y_hour) == 0:
                continue
            x_hour = x_hour.sort_values("time_in_hour").tail(input_window)
            y_hour = y_hour.sort_values("time_in_hour")

            x_arr = _to_float32(x_hour, self.feature_cols)  # [T, F]
            targets: List[np.ndarray] = []
            if "midprice" in self.target_heads:
                targets.append(_to_float32(y_hour, ["mid_price" if "mid_price" in y_hour.columns else "midprice"]))
            if "spread" in self.target_heads:
                targets.append(_to_float32(y_hour, ["spread"]))
            if "top_liquidity" in self.target_heads:
                # top-of-book liquidity: ask_vol_1 + bid_vol_1
                liq = (
                    y_hour["ask_vol_1"].fillna(0.0).astype(np.float32).to_numpy()
                    + y_hour["bid_vol_1"].fillna(0.0).astype(np.float32).to_numpy()
                )
                targets.append(liq[:, None])
            y_arr = np.concatenate(targets, axis=1)  # [60, H]
            self.samples.append((x_arr, y_arr))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# -----------------------------
# Model
# -----------------------------

class CausalConvBlock(nn.Module):
    """Residual causal 1D conv block with dilation (TCN-style)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = None
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self._padding:
            out = out[..., :-self._padding]
        return out + x


class MultiHorizonTCN(nn.Module):
    """
    Encoder: causal TCN over 59-minute history.
    Decoder head: linear projection to H targets per horizon step (60 seconds).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.05,
        horizon: int = 60,
        head_dim: int = len(TARGET_HEADS),
    ):
        super().__init__()
        self.proj_in = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        blocks = []
        for i in range(num_layers):
            dilation = 2**i
            blocks.append(CausalConvBlock(hidden_dim, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_dim, horizon * head_dim, kernel_size=1)
        self.horizon = horizon
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        returns: [B, horizon, head_dim]
        """
        x = x.transpose(1, 2)  # [B, F, T]
        h = self.proj_in(x)
        h = self.tcn(h)
        logits = self.head(h)  # [B, horizon*H, T]
        # use the last timestep (causal); shape -> [B, horizon, H]
        last = logits[..., -1]
        return last.view(x.size(0), self.horizon, self.head_dim)


# -----------------------------
# Loss
# -----------------------------

def multi_horizon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    head_weights: Sequence[float],
    smooth_lambda: float = 0.02,
) -> torch.Tensor:
    """
    pred/target: [B, 60, H]
    head_weights: length H
    smoothness: penalize successive diffs on midprice (head 0 assumed mid).
    """
    mse = (pred - target) ** 2
    weights = torch.tensor(head_weights, device=pred.device, dtype=pred.dtype)
    mse = mse * weights
    loss = mse.mean()

    # Smoothness on head 0 (midprice)
    if smooth_lambda > 0 and pred.size(-1) > 0:
        mid = pred[..., 0]
        smooth = (mid[:, 1:] - mid[:, :-1]) ** 2
        loss = loss + smooth_lambda * smooth.mean()
    return loss


# -----------------------------
# Scheduler (forecast -> positions)
# -----------------------------

@dataclass
class ScheduleConfig:
    window: int = 14
    alpha: float = 0.3  # blend weight for model vs fixed TWAP in window
    price_cap: float = 3.0  # cap on price_score multiple vs mean
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
    """
    Inputs are forecasts only (no real last-minute data).
    Returns a 60-length positions array summing to volume_to_fill.
    """
    assert mid_pred.shape[0] == 60, "Need 60-second forecast for the last minute."
    p_close = mid_pred[-1]

    # Directional: favor seconds where predicted price is below close (cheaper to buy)
    favorability = p_close - mid_pred
    temp = np.std(favorability) + cfg.eps
    price_score = np.exp(np.clip(favorability / temp, -cfg.price_cap, cfg.price_cap))

    liq_score = np.ones_like(price_score)
    if liq_pred is not None:
        mean_liq = np.mean(liq_pred) + cfg.eps
        liq_score = np.clip(liq_pred / mean_liq, cfg.liq_min, cfg.liq_max)

    spread_score = np.ones_like(price_score)
    if spread_pred is not None:
        mean_spread = np.mean(spread_pred) + cfg.eps
        spread_score = np.clip(mean_spread / (spread_pred + cfg.eps), cfg.spread_min, cfg.spread_max)

    score = price_score * liq_score * spread_score

    # Restrict to end-of-minute window for robustness.
    start = 60 - cfg.window
    mask = np.zeros_like(score)
    mask[start:] = 1.0
    score = score * mask

    if score.sum() <= 0:
        # Fallback: fixed TWAP in the window.
        alloc = np.zeros(60, dtype=np.float32)
        alloc[start:] = volume_to_fill / cfg.window
        return alloc

    weights = score / score.sum()
    model_sched = weights * volume_to_fill

    # Blend with fixed TWAP in the same window for stability.
    twap = np.zeros(60, dtype=np.float32)
    twap[start:] = volume_to_fill / cfg.window
    blended = cfg.alpha * model_sched + (1.0 - cfg.alpha) * twap
    # Normalize to exact volume_to_fill.
    blended = blended * (volume_to_fill / blended.sum())
    return blended.astype(np.float32)


# -----------------------------
# Training / Evaluation helpers
# -----------------------------

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    smooth_lambda: float = 0.02
    head_weights: Tuple[float, float, float] = (1.0, 0.2, 0.2)  # mid, spread, liq
    device: str = "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(cfg.device)
        yb = yb.to(cfg.device)
        optim.zero_grad()
        pred = model(xb)
        loss = multi_horizon_loss(pred, yb, cfg.head_weights, smooth_lambda=cfg.smooth_lambda)
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def evaluate_mse(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            pred = model(xb)
            loss = multi_horizon_loss(pred, yb, cfg.head_weights, smooth_lambda=cfg.smooth_lambda)
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


def predict_paths(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> List[np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(cfg.device)
            pred = model(xb).cpu().numpy()
            preds.extend(pred)
    return preds


# -----------------------------
# Example (wiring only; not executed automatically)
# -----------------------------
if __name__ == "__main__":
    print(
        "This module defines a leakage-safe multi-horizon forecaster and "
        "scheduler. Import the classes/functions and wire them into your "
        "training script. Example usage:\n"
        "  - Build features for X (0-58:59) and Y (59:00-59:59).\n"
        "  - ds = LastMinuteDataset(x_df, y_df)\n"
        "  - model = MultiHorizonTCN(input_dim=len(DEFAULT_FEATURES))\n"
        "  - Train with TrainConfig; evaluate by converting forecasts to schedules\n"
        "    using build_schedule_from_forecasts and scoring via simulate_walk_the_book."
    )




