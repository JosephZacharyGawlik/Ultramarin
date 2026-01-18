#!/usr/bin/env python3
"""
Temporal Convolutional Network baseline for last-minute execution forecasting.

Mirrors the ridge pipeline but replaces the linear regressor with a PyTorch TCN
that consumes the full 59-minute per-second history.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import os
import sys
from pathlib import Path

# Ensure sandbox-friendly OpenMP settings even when running with the system python.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_CREATE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Torch is installed inside .venv; add that site-packages to sys.path when needed.
REPO_ROOT = Path(__file__).resolve().parent
VENV_SITE = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if str(VENV_SITE) not in sys.path and VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(1)

from simulate_walk_the_book import simulate_walk_the_book

INPUT_WINDOW = 300  # seconds to look back (last 5 minutes)
SEQ_LEN = INPUT_WINDOW
FEATURE_COLUMNS = [
    "midprice",
    "spread",
    "spread_pct",
    "ask_price_1",
    "bid_price_1",
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
# Per-second targets for 59:00-59:59 (midprice only)
TARGET_PER_SECOND = ["midprice"]
TARGET_WEIGHTS = np.array([1.0], dtype=np.float32)
TARGET_SIZE = 60 * len(TARGET_PER_SECOND)


def create_submission_template(y_df: pd.DataFrame) -> pd.DataFrame:
    ids = np.sort(y_df["anonymized_id"].unique())
    times = np.sort(y_df["time_in_hour"].unique())
    records = [
        {"anonymized_id": int(id_), "time_in_hour": ts}
        for id_ in ids
        for ts in times
    ]
    return pd.DataFrame(records)


def baseline_uniform_submission(y_df: pd.DataFrame, volume_to_fill: float) -> pd.DataFrame:
    template = create_submission_template(y_df)
    template["position"] = volume_to_fill / 60.0
    return template


def baseline_end_of_hour_submission(
    y_df: pd.DataFrame, volume_to_fill: float, seconds_from_end: int
) -> pd.DataFrame:
    template = create_submission_template(y_df)
    seconds = (
        (template["time_in_hour"] - pd.Timedelta(minutes=59))
        .dt.total_seconds()
        .astype(int)
    )
    cutoff = 60 - seconds_from_end
    template["position"] = np.where(
        seconds >= cutoff, volume_to_fill / seconds_from_end, 0.0
    )
    return template


def baseline_liquidity_driven_submission(
    y_df: pd.DataFrame,
    volume_to_fill: float,
    seconds_from_end: int = 14,
    min_multiplier: float = 0.5,
    max_multiplier: float = 1.5,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for hour_id, hour_df in y_df.groupby("anonymized_id"):
        hour_sorted = hour_df.sort_values("time_in_hour").reset_index(drop=True)
        seconds = (
            (hour_sorted["time_in_hour"] - pd.Timedelta(minutes=59))
            .dt.total_seconds()
            .astype(int)
        )
        positions: List[float] = []
        remaining = float(volume_to_fill)
        window_rows_seen = 0
        prev_liquidity: float | None = None
        window_start = 60 - seconds_from_end

        for sec, (_, row) in zip(seconds, hour_sorted.iterrows()):
            ask_vol = float(row["ask_vol_1"]) if pd.notna(row["ask_vol_1"]) else 0.0
            bid_vol = float(row["bid_vol_1"]) if pd.notna(row["bid_vol_1"]) else 0.0
            current_liquidity = ask_vol + bid_vol
            in_window = sec >= window_start

            if not in_window or remaining <= 0:
                positions.append(0.0)
                continue

            remaining_slots = max(seconds_from_end - window_rows_seen, 1)
            base_allocation = remaining / remaining_slots

            if prev_liquidity is None:
                liquidity_ratio = 1.0
            elif prev_liquidity == 0:
                liquidity_ratio = (
                    max_multiplier if current_liquidity > 0 else min_multiplier
                )
            else:
                liquidity_ratio = (
                    current_liquidity / prev_liquidity if prev_liquidity > 0 else 1.0
                )

            multiplier = float(
                np.clip(liquidity_ratio, min_multiplier, max_multiplier)
            )
            position = min(base_allocation * multiplier, remaining)

            positions.append(position)
            remaining -= position
            window_rows_seen += 1
            prev_liquidity = current_liquidity

        if window_rows_seen == 0:
            positions = [0.0 for _ in positions]

        hour_records = [
            {
                "anonymized_id": hour_id,
                "time_in_hour": time_value,
                "position": float(pos),
            }
            for time_value, pos in zip(hour_sorted["time_in_hour"], positions)
        ]
        records.extend(hour_records)

    return pd.DataFrame(records)


def compute_implementation_error(
    submission_df: pd.DataFrame,
    y_df: pd.DataFrame,
    volume_to_fill: float,
) -> float:
    merged = submission_df.merge(
        y_df, on=["anonymized_id", "time_in_hour"], how="left", suffixes=("", "_y")
    ).sort_values(["anonymized_id", "time_in_hour"])

    errors: List[float] = []
    for _, hour_df in merged.groupby("anonymized_id"):
        close_series = hour_df["close"].dropna()
        if close_series.empty:
            continue
        close_price = float(close_series.iloc[-1])
        positions = hour_df["position"].fillna(0.0).to_numpy(dtype=float)

        ask_prices = np.column_stack(
            [hour_df[f"ask_price_{lvl}"].to_numpy(dtype=float) for lvl in range(1, 6)]
        )
        ask_vols = np.column_stack(
            [
                hour_df[f"ask_vol_{lvl}"].fillna(0.0).to_numpy(dtype=float)
                for lvl in range(1, 6)
            ]
        )
        bid_prices = np.column_stack(
            [hour_df[f"bid_price_{lvl}"].to_numpy(dtype=float) for lvl in range(1, 6)]
        )
        bid_vols = np.column_stack(
            [
                hour_df[f"bid_vol_{lvl}"].fillna(0.0).to_numpy(dtype=float)
                for lvl in range(1, 6)
            ]
        )

        total_vol, avg_price = simulate_walk_the_book(
            positions, ask_prices, ask_vols, bid_prices, bid_vols
        )
        if total_vol <= 0 or np.isnan(avg_price) or np.isnan(close_price):
            continue

        relative_error = abs(avg_price - close_price) / close_price
        fill_penalty = min(100.0, volume_to_fill / total_vol)
        errors.append(relative_error * fill_penalty * 10000.0)

    return float(np.mean(errors)) if errors else float("nan")


@dataclass
class SequenceSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_ids: np.ndarray
    val_ids: np.ndarray
    val_meta: pd.DataFrame
    target_mean: np.ndarray
    target_std: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + residual)


class TCNRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 5,
        num_layers: int = 3,
        dropout: float = 0.1,
        target_dim: int = TARGET_SIZE,
    ) -> None:
        super().__init__()
        layers = []
        channels = input_channels
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )
            channels = hidden_channels

        self.network = nn.Sequential(*layers)
        self.regressor = nn.Linear(hidden_channels, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # batch, channels, seq_len
        out = self.network(x)
        out = out[:, :, -1]  # last time step
        return self.regressor(out)


def load_symbol_frames(root: Path, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = root / symbol
    x_path = base / "X_train.parquet"
    y_path = base / "y_train.parquet"
    x_df = pd.read_parquet(x_path)
    y_df = pd.read_parquet(y_path)
    x_df = x_df.sort_values(["anonymized_id", "time_in_hour"]).reset_index(drop=True)
    y_df = y_df.sort_values(["anonymized_id", "time_in_hour"]).reset_index(drop=True)
    return x_df, y_df


def enrich_x_columns(x_df: pd.DataFrame) -> pd.DataFrame:
    df = x_df.copy()
    df["second_of_hour"] = (df["time_in_hour"] / pd.Timedelta(seconds=1)).astype(int)
    df["midprice"] = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    bid_cols = [f"bid_vol_{lvl}" for lvl in range(1, 6)]
    ask_cols = [f"ask_vol_{lvl}" for lvl in range(1, 6)]
    df["bid_depth_total"] = df[bid_cols].sum(axis=1)
    df["ask_depth_total"] = df[ask_cols].sum(axis=1)
    df["total_depth"] = df["bid_depth_total"] + df["ask_depth_total"]
    df["imbalance"] = (df["bid_depth_total"] - df["ask_depth_total"]) / (
        df["total_depth"] + 1e-9
    )
    df["volume"] = df["volume"].fillna(0.0)
    df["midprice"] = df.groupby("anonymized_id")["midprice"].ffill()
    df["midprice"] = df.groupby("anonymized_id")["midprice"].bfill()
    df["spread_pct"] = df["spread"] / df["midprice"].replace(0, np.nan)

    df["log_mid"] = np.log(df["midprice"].replace(0, np.nan))
    df["mid_return_1s"] = (
        df.groupby("anonymized_id")["log_mid"].diff().fillna(0.0)
    )

    df["volume_roll_mean_5"] = (
        df.groupby("anonymized_id")["volume"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        .fillna(0.0)
    )
    df["spread_roll_mean_5"] = (
        df.groupby("anonymized_id")["spread"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        .fillna(0.0)
    )
    df["imbalance_roll_mean_5"] = (
        df.groupby("anonymized_id")["imbalance"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        .fillna(0.0)
    )
    df = df.drop(columns=["log_mid"])
    return df


def build_targets(y_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    df = y_df.copy()
    df["midprice"] = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    for col in TARGET_PER_SECOND:
        df[col] = df[col].fillna(0.0)

    target_map: Dict[int, np.ndarray] = {}
    for hour_id, group in df.groupby("anonymized_id"):
        group_sorted = group.sort_values("time_in_hour")
        arr = group_sorted[TARGET_PER_SECOND].to_numpy(dtype=np.float32)
        # Ensure 60 seconds
        if arr.shape[0] != 60:
            if arr.shape[0] < 60:
                pad = np.zeros((60 - arr.shape[0], len(TARGET_PER_SECOND)), dtype=np.float32)
                arr = np.vstack([arr, pad])
            else:
                arr = arr[:60]
        target_map[int(hour_id)] = arr
    return target_map


def build_sequence_dataset(
    x_df: pd.DataFrame, y_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    enriched = enrich_x_columns(x_df)
    target_map = build_targets(y_df)

    seq_list: List[np.ndarray] = []
    meta_rows: List[Dict[str, float]] = []
    targets_rows: List[np.ndarray] = []

    feature_cols = FEATURE_COLUMNS
    enriched[feature_cols] = enriched[feature_cols].fillna(0.0)

    for hour_id, group in enriched.groupby("anonymized_id"):
        hour_id_int = int(hour_id)
        if hour_id_int not in target_map:
            continue
        group_sorted = group.sort_values("time_in_hour")
        # take only last INPUT_WINDOW seconds before 59:00
        cutoff_time = pd.Timedelta(minutes=59) - pd.Timedelta(seconds=INPUT_WINDOW)
        window_group = group_sorted[group_sorted["time_in_hour"] >= cutoff_time]
        seq = np.zeros((SEQ_LEN, len(feature_cols)), dtype=np.float32)
        # map seconds to index relative to window
        seconds = (
            (window_group["time_in_hour"] - cutoff_time)
            .dt.total_seconds()
            .astype(int)
            .to_numpy()
        )
        sec_mask = (seconds >= 0) & (seconds < SEQ_LEN)
        seconds = seconds[sec_mask]
        seq[seconds, :] = window_group[feature_cols].to_numpy(dtype=np.float32)[sec_mask]
        seq_list.append(seq)
        meta_rows.append({"anonymized_id": hour_id_int})
        targets_rows.append(target_map[hour_id_int])

    if not seq_list:
        raise ValueError("No sequences built. Check data alignment.")

    sequences = np.stack(seq_list)

    feature_mean = sequences.mean(axis=(0, 1), keepdims=True)
    feature_std = sequences.std(axis=(0, 1), keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    normalized_sequences = (sequences - feature_mean) / feature_std

    targets_array = np.stack(targets_rows)  # shape (N, 60, len(TARGET_PER_SECOND))
    targets_array = targets_array.reshape(targets_array.shape[0], -1)  # flatten

    meta_frame = pd.DataFrame(meta_rows)
    return normalized_sequences, targets_array, meta_frame


def train_val_split_sequences(
    sequences: np.ndarray,
    targets: np.ndarray,
    meta: pd.DataFrame,
    val_fraction: float,
) -> SequenceSplit:
    ids = meta["anonymized_id"].to_numpy()
    unique_ids = ids.copy()
    split_idx = max(1, int(len(unique_ids) * (1 - val_fraction)))
    train_mask = np.isin(ids, unique_ids[:split_idx])
    val_mask = ~train_mask

    target_mean = targets[train_mask].mean(axis=0)
    target_std = targets[train_mask].std(axis=0)
    target_std[target_std < 1e-6] = 1.0

    return SequenceSplit(
        X_train=sequences[train_mask],
        y_train=(targets[train_mask] - target_mean) / target_std,
        X_val=sequences[val_mask],
        y_val=(targets[val_mask] - target_mean) / target_std,
        train_ids=ids[train_mask],
        val_ids=ids[val_mask],
        val_meta=meta[val_mask].reset_index(drop=True),
        target_mean=target_mean,
        target_std=target_std,
    )


def train_tcn(
    train_data: SequenceDataset,
    val_data: SequenceDataset,
    epochs: int = 15,
    lr: float = 1e-3,
    smooth_lambda: float = 0.02,
) -> Tuple[TCNRegressor, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNRegressor(input_channels=train_data.X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = torch.from_numpy(TARGET_WEIGHTS).to(device)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    history: List[float] = []

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            preds_seq = preds.view(preds.shape[0], 60, len(TARGET_PER_SECOND))
            target_seq = batch_y.view(batch_y.shape[0], 60, len(TARGET_PER_SECOND))
            mse = (preds_seq - target_seq) ** 2
            diff = preds_seq[:, 1:, :] - preds_seq[:, :-1, :]
            smooth = (diff**2).mean()
            loss = (mse * weights).mean() + smooth_lambda * smooth
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_X)
                preds_seq = preds.view(preds.shape[0], 60, len(TARGET_PER_SECOND))
                target_seq = batch_y.view(batch_y.shape[0], 60, len(TARGET_PER_SECOND))
                mse = (preds_seq - target_seq) ** 2
                diff = preds_seq[:, 1:, :] - preds_seq[:, :-1, :]
                smooth = (diff**2).mean()
                val_loss = ((mse * weights).mean() + smooth_lambda * smooth).item()
                val_losses.append(val_loss)
        epoch_loss = float(np.mean(val_losses))
        history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Validation MSE: {epoch_loss:.6f}")

    return model, history


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channels: List[str],
) -> Dict[str, Dict[str, float]]:
    # y_true/y_pred shape: (N, 60, C)
    metrics: Dict[str, Dict[str, float]] = {}
    for c_idx, name in enumerate(channels):
        true_c = y_true[:, :, c_idx].reshape(-1)
        pred_c = y_pred[:, :, c_idx].reshape(-1)
        errors = true_c - pred_c
        ss_res = float(np.sum(errors**2))
        centered = true_c - np.mean(true_c)
        ss_tot = float(np.sum(centered**2)) or 1.0
        metrics[name] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "r2": 1.0 - ss_res / ss_tot,
        }
    return metrics


def score_seconds(pred_seq: np.ndarray) -> np.ndarray:
    """
    pred_seq shape: (60, 1) for midprice
    Returns per-second scores to weight allocations.
    """
    mid = pred_seq[:, 0]
    close_est = mid[-1] if mid[-1] > 0 else np.nanmean(mid)
    if np.isnan(close_est) or close_est <= 0:
        close_est = np.nanmean(mid[mid > 0]) if np.any(mid > 0) else 1.0

    price_gap = np.abs(mid - close_est) / close_est
    price_score = np.exp(-5 * price_gap)  # dominant
    raw = price_score
    raw[np.isnan(raw)] = 0.0
    return raw


def allocate_positions_from_preds(
    pred_seq: np.ndarray, volume_to_fill: float
) -> np.ndarray:
    scores = score_seconds(pred_seq)
    # cap extreme scores to avoid over-concentration
    if scores.sum() > 0:
        avg = scores.mean()
        scores = np.clip(scores, 0, 3.0 * avg)
    total_score = scores.sum()
    if total_score <= 0:
        return np.full(60, volume_to_fill / 60.0, dtype=float)
    weights = scores / total_score
    return weights * volume_to_fill


def generate_submission_from_predictions(
    pred_array: np.ndarray,
    val_ids: np.ndarray,
    y_df: pd.DataFrame,
    volume_to_fill: float,
    blend_alpha: float = 0.2,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    pred_array = pred_array.reshape(pred_array.shape[0], 60, len(TARGET_PER_SECOND))
    smart_submission = baseline_liquidity_driven_submission(
        y_df, volume_to_fill, seconds_from_end=14
    )
    smart_key = (
        smart_submission[["anonymized_id", "time_in_hour", "position"]]
        .rename(columns={"position": "smart_pos"})
    )
    for hour_idx, hour_id in enumerate(val_ids):
        pred_seq = pred_array[hour_idx]
        model_positions = allocate_positions_from_preds(pred_seq, volume_to_fill)
        hour_times = (
            y_df[y_df["anonymized_id"] == hour_id]
            .sort_values("time_in_hour")["time_in_hour"]
            .to_numpy()
        )
        if len(hour_times) != 60:
            continue
        temp_df = pd.DataFrame(
            {
                "anonymized_id": hour_id,
                "time_in_hour": hour_times,
                "model_pos": model_positions,
            }
        )
        temp_df = temp_df.merge(smart_key, on=["anonymized_id", "time_in_hour"], how="left")
        temp_df["smart_pos"] = temp_df["smart_pos"].fillna(0.0)
        temp_df["blended"] = (
            blend_alpha * temp_df["model_pos"] + (1 - blend_alpha) * temp_df["smart_pos"]
        )
        # rescale to exact volume_to_fill
        total = temp_df["blended"].sum()
        if total <= 0:
            temp_df["blended"] = volume_to_fill / 60.0
        else:
            temp_df["blended"] = temp_df["blended"] * (volume_to_fill / total)
        for _, row in temp_df.iterrows():
            records.append(
                {
                    "anonymized_id": hour_id,
                    "time_in_hour": row["time_in_hour"],
                    "position": float(row["blended"]),
                }
            )
    return pd.DataFrame(records)


def generate_predicted_submission(
    y_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    volume_to_fill: float,
) -> pd.DataFrame:
    raise NotImplementedError("Deprecated in per-second target version")


def evaluate_predicted_strategy(
    predictions: np.ndarray,
    val_ids: np.ndarray,
    val_meta: pd.DataFrame,
    y_df: pd.DataFrame,
    volume_to_fill: float,
) -> Tuple[float, float, float, float]:
    y_val_df = y_df[y_df["anonymized_id"].isin(val_ids)].copy()
    submission_df = generate_submission_from_predictions(
        predictions, val_ids, y_val_df, volume_to_fill
    )

    if submission_df.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")

    print("\nSample TCN positions (first 3 validation hours):")
    sampled_ids = submission_df["anonymized_id"].unique()[:3]
    for hour_id in sampled_ids:
        sample = submission_df[submission_df["anonymized_id"] == hour_id]
        if sample.empty:
            continue
        sample = sample.sort_values("time_in_hour")
        seconds = (
            (sample["time_in_hour"] - pd.Timedelta(minutes=59))
            .dt.total_seconds()
            .astype(int)
        )
        positions = np.round(sample["position"].to_numpy(dtype=float), 4)
        print(f"Hour {hour_id}:")
        for sec, pos in zip(seconds, positions):
            print(f"  t+{sec:02d}s -> {pos}")
        print("---")

    model_error = compute_implementation_error(
        submission_df, y_val_df, volume_to_fill
    )

    uniform_submission = baseline_uniform_submission(y_val_df, volume_to_fill)
    end_submission = baseline_end_of_hour_submission(
        y_val_df, volume_to_fill, seconds_from_end=20
    )
    smart_submission = baseline_liquidity_driven_submission(
        y_val_df, volume_to_fill, seconds_from_end=14
    )

    uniform_error = compute_implementation_error(
        uniform_submission, y_val_df, volume_to_fill
    )
    end_error = compute_implementation_error(
        end_submission, y_val_df, volume_to_fill
    )
    smart_error = compute_implementation_error(
        smart_submission, y_val_df, volume_to_fill
    )

    return model_error, uniform_error, end_error, smart_error


def read_volume_to_fill(data_root: Path, symbol: str) -> float:
    vol_path = data_root / symbol / "vol_to_fill.txt"
    text = vol_path.read_text().strip()
    for token in text.replace(":", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    raise ValueError(f"Could not parse volume from {vol_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TCN baseline for execution targets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Root directory containing the symbol folders.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol folder to use (e.g., BTCUSDT).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of hours reserved for validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Training epochs for the TCN.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the TCN optimizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_df, y_df = load_symbol_frames(args.data_root, args.symbol)
    sequences, targets, meta = build_sequence_dataset(x_df, y_df)
    split = train_val_split_sequences(
        sequences, targets, meta, val_fraction=args.val_fraction
    )

    train_dataset = SequenceDataset(split.X_train, split.y_train)
    val_dataset = SequenceDataset(split.X_val, split.y_val)
    model, _ = train_tcn(
        train_dataset, val_dataset, epochs=args.epochs, lr=args.lr
    )

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(split.X_val).float().to(device)
        predictions = model(val_tensor).cpu().numpy()

    denormalized_preds = predictions * split.target_std + split.target_mean
    denormalized_preds = denormalized_preds.reshape(
        denormalized_preds.shape[0], 60, len(TARGET_PER_SECOND)
    )
    denormalized_targets = split.y_val * split.target_std + split.target_mean
    denormalized_targets = denormalized_targets.reshape(
        denormalized_targets.shape[0], 60, len(TARGET_PER_SECOND)
    )

    metrics = compute_metrics(
        denormalized_targets, denormalized_preds, TARGET_PER_SECOND
    )
    print("\nValidation metrics (per-second targets flattened):")
    for name, stats in metrics.items():
        print(
            f"{name:12s} | MAE: {stats['mae']:.6f}, RMSE: {stats['rmse']:.6f}, R2: {stats['r2']:.4f}"
        )

    volume_to_fill = read_volume_to_fill(args.data_root, args.symbol)
    model_error, uniform_error, end_error, smart_error = evaluate_predicted_strategy(
        denormalized_preds,
        split.val_ids,
        split.val_meta,
        y_df,
        volume_to_fill,
    )
    print("\nImplementation error (validation subset):")
    print(f"TCN-guided strategy: {model_error:.2f} bps")
    print(f"Uniform baseline:    {uniform_error:.2f} bps")
    print(f"End-of-hour baseline:{end_error:.2f} bps")
    print(f"Liquidity baseline:  {smart_error:.2f} bps")


if __name__ == "__main__":
    main()
