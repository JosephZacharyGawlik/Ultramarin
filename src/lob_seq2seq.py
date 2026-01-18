#!/usr/bin/env python3
"""
Leakage-safe LOB encoder + seq2seq attention decoder for last-minute forecasting.

This follows the spirit of DeepLOB (spatial+temporal encoder) and multi-horizon
forecasting (decoder predicts an entire 60-step path), without any TCNs.
Only first 59 minutes are used as input; the last minute is never observed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Sandbox-friendly threading defaults.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_CREATE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# Discover local venv if present.
REPO_ROOT = Path(__file__).resolve().parent
VENV_SITE = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if str(VENV_SITE) not in sys.path and VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))


# -----------------------------
# Data
# -----------------------------

FEATURE_COLS: List[str] = [
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


class LastMinuteSeq2SeqDataset(Dataset):
    """
    Builds (X_history, y_future) pairs per hour.
      X: 59 minutes of per-second features -> [T, F]
      y: 60-second midprice path -> [60]
    """

    def __init__(
        self,
        x_df: pd.DataFrame,
        y_df: pd.DataFrame,
        feature_cols: Sequence[str] = FEATURE_COLS,
        input_window: int = 59 * 60,
    ):
        self.feature_cols = list(feature_cols)
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for hour_id, x_hour in x_df.groupby("anonymized_id"):
            y_hour = y_df[y_df["anonymized_id"] == hour_id]
            if len(y_hour) == 0:
                continue
            x_hour = x_hour.sort_values("time_in_hour")
            y_hour = y_hour.sort_values("time_in_hour")

            # Enforce fixed input_window length: pad at the front if needed, else tail.
            x_arr = x_hour[self.feature_cols].astype(np.float32).to_numpy()
            if x_arr.shape[0] < input_window:
                pad_len = input_window - x_arr.shape[0]
                pad = np.zeros((pad_len, x_arr.shape[1]), dtype=np.float32)
                x_arr = np.vstack([pad, x_arr])
            elif x_arr.shape[0] > input_window:
                x_arr = x_arr[-input_window:, :]

            mid = ((y_hour["ask_price_1"] + y_hour["bid_price_1"]) / 2.0).astype(np.float32).to_numpy()
            if mid.shape[0] != 60:
                continue  # enforce full last-minute availability
            if not np.isfinite(x_arr).all() or not np.isfinite(mid).all():
                continue
            self.samples.append((x_arr, mid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# -----------------------------
# Model (DeepLOB-inspired encoder + attention decoder)
# -----------------------------


class SpatialBlock(nn.Module):
    """
    Lightweight spatial encoder over feature channels.
    Not a full price-level image, but 1D conv can still learn local mixes.
    """

    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        h = self.net(x)
        return h.transpose(1, 2)  # [B, T, H]


class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W_e = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_d = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_outputs: torch.Tensor, dec_state: torch.Tensor) -> torch.Tensor:
        # enc_outputs: [B, T, E], dec_state: [B, D]
        e = self.W_e(enc_outputs) + self.W_d(dec_state).unsqueeze(1)  # [B, T, A]
        scores = self.v(torch.tanh(e)).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * enc_outputs, dim=1)  # [B, E]
        return context


class LOBSeq2Seq(nn.Module):
    """
    Encoder: spatial block -> bidirectional LSTM over 59-minute sequence.
    Decoder: autoregressive LSTM with additive attention; predicts 60-step midprice.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 128,
        enc_layers: int = 1,
        dec_layers: int = 1,
        dropout: float = 0.1,
        horizon: int = 60,
    ):
        super().__init__()
        self.spatial = SpatialBlock(feature_dim, hidden)
        self.encoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=enc_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )
        self.attn = AdditiveAttention(enc_dim=2 * hidden, dec_dim=hidden)
        self.decoder = nn.LSTM(
            input_size=1 + 2 * hidden,  # previous y + context
            hidden_size=hidden,
            num_layers=dec_layers,
            batch_first=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, 1)
        self.init_h = nn.Linear(2 * hidden, hidden)
        self.init_c = nn.Linear(2 * hidden, hidden)
        self.horizon = horizon

    def forward(self, x: torch.Tensor, y_teacher: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, T, F]
        y_teacher: [B, 60] if provided (teacher forcing), else autoregressive
        returns: [B, 60]
        """
        B = x.size(0)
        h_spatial = self.spatial(x)  # [B, T, H]
        enc_out, _ = self.encoder(h_spatial)  # [B, T, 2H]

        # Init decoder state from encoder mean.
        enc_mean = enc_out.mean(dim=1)
        dec_h0 = torch.tanh(self.init_h(enc_mean))
        dec_c0 = torch.tanh(self.init_c(enc_mean))
        dec_h = dec_h0.unsqueeze(0)  # [1, B, H]
        dec_c = dec_c0.unsqueeze(0)

        outputs: List[torch.Tensor] = []
        # Seed: use last encoder input midprice proxy (x[...,0]) mean over last few steps
        prev_y = x[:, -1, 0].unsqueeze(-1)  # [B, 1]

        for t in range(self.horizon):
            context = self.attn(enc_out, dec_h[-1])  # [B, 2H]
            dec_in = torch.cat([prev_y, context], dim=1).unsqueeze(1)  # [B, 1, 1+2H]
            dec_out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            y_hat = self.out(dec_out).squeeze(1)  # [B, 1]
            outputs.append(y_hat.squeeze(-1))
            if self.training and y_teacher is not None:
                prev_y = y_teacher[:, t].unsqueeze(-1)
            else:
                prev_y = y_hat.detach()

        return torch.stack(outputs, dim=1)  # [B, 60]


# -----------------------------
# Loss
# -----------------------------

def forecast_loss(pred: torch.Tensor, target: torch.Tensor, smooth_lambda: float = 0.02) -> torch.Tensor:
    mse = (pred - target) ** 2
    loss = mse.mean()
    if smooth_lambda > 0:
        smooth = (pred[:, 1:] - pred[:, :-1]) ** 2
        loss = loss + smooth_lambda * smooth.mean()
    return loss


# -----------------------------
# Scheduler (forecast -> positions, no leakage)
# -----------------------------

@dataclass
class ScheduleCfg:
    window: int = 14
    price_cap: float = 3.0
    alpha: float = 0.3  # blend with fixed TWAP in window
    eps: float = 1e-6


def schedule_from_mid_forecast(mid_pred: np.ndarray, volume_to_fill: float, cfg: ScheduleCfg = ScheduleCfg()) -> np.ndarray:
    assert mid_pred.shape[0] == 60
    if not np.isfinite(mid_pred).all():
        # Fallback to fixed TWAP in window if forecast is invalid.
        twap = np.zeros(60, dtype=np.float32)
        twap[60 - cfg.window :] = volume_to_fill / cfg.window
        return twap
    p_close = mid_pred[-1]
    price_score = 1.0 / (np.abs(mid_pred - p_close) + cfg.eps)
    price_score = np.minimum(price_score, cfg.price_cap * price_score.mean())

    mask = np.zeros_like(price_score)
    start = 60 - cfg.window
    mask[start:] = 1.0
    score = price_score * mask

    if score.sum() <= 0:
        alloc = np.zeros(60, dtype=np.float32)
        alloc[start:] = volume_to_fill / cfg.window
        return alloc

    weights = score / score.sum()
    model_sched = weights * volume_to_fill

    twap = np.zeros(60, dtype=np.float32)
    twap[start:] = volume_to_fill / cfg.window
    blended = cfg.alpha * model_sched + (1.0 - cfg.alpha) * twap
    blended = blended * (volume_to_fill / blended.sum())
    return blended.astype(np.float32)


# -----------------------------
# Training utilities (minimal)
# -----------------------------

@dataclass
class TrainCfg:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    smooth_lambda: float = 0.02
    device: str = "cpu"


def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, cfg: TrainCfg) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(cfg.device)
        yb = yb.to(cfg.device)
        opt.zero_grad()
        pred = model(xb, y_teacher=yb)
        loss = forecast_loss(pred, yb, smooth_lambda=cfg.smooth_lambda)
        loss.backward()
        opt.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def eval_epoch(model: nn.Module, loader: DataLoader, cfg: TrainCfg) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            pred = model(xb, y_teacher=None)
            loss = forecast_loss(pred, yb, smooth_lambda=cfg.smooth_lambda)
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


if __name__ == "__main__":
    print(
        "This module defines a non-TCN, leakage-safe encoder-decoder with attention.\n"
        "Wire it into your training script as follows:\n"
        "  ds = LastMinuteSeq2SeqDataset(x_df, y_df)\n"
        "  model = LOBSeq2Seq(feature_dim=len(FEATURE_COLS))\n"
        "  Train with teacher forcing (train_epoch/eval_epoch)\n"
        "  Forecast 60s midpath, then build positions via schedule_from_mid_forecast\n"
        "  Evaluate schedules with simulate_walk_the_book.\n"
    )

