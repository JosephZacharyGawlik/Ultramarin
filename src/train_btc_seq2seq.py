#!/usr/bin/env python3
"""
End-to-end, leakage-safe seq2seq training + evaluation on BTCUSDT.

Pipeline:
 1) Load BTCUSDT X_train/y_train parquet and vol_to_fill.
 2) Feature-engineer the first 59 minutes (no last-minute leakage).
 3) Split hours chronologically into train/val.
 4) Train the attention seq2seq model from lob_seq2seq.py (no TCNs).
 5) Forecast the last-minute midprice path for val hours.
 6) Convert forecasts to execution schedules (no last-minute data used).
 7) Score with simulate_walk_the_book.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lob_seq2seq import (
    FEATURE_COLS,
    LastMinuteSeq2SeqDataset,
    LOBSeq2Seq,
    TrainCfg,
    eval_epoch,
    schedule_from_mid_forecast,
    train_epoch,
)
from simulate_walk_the_book import simulate_walk_the_book

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parent
BTC_DIR = REPO_ROOT.parent / "data" / "BTCUSDT"


# -----------------------------
# Data loading / features
# -----------------------------

def load_btc_frames() -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    x_path = BTC_DIR / "X_train.parquet"
    y_path = BTC_DIR / "y_train.parquet"
    vol_path = BTC_DIR / "vol_to_fill.txt"

    x_df = pd.read_parquet(x_path).sort_values(["anonymized_id", "time_in_hour"])
    y_df = pd.read_parquet(y_path).sort_values(["anonymized_id", "time_in_hour"])

    with open(vol_path, "r") as f:
        txt = f.read().strip()
    m = re.search(r"The volume to fill is: ([\d.]+)", txt)
    volume_to_fill = float(m.group(1)) if m else None

    return x_df, y_df, volume_to_fill


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
    # Replace any remaining NaNs/Infs in feature columns
    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def standardize(train_df: pd.DataFrame, full_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    full_df = full_df.copy()
    for c in cols:
        mean = train_df[c].mean()
        std = train_df[c].std(ddof=0) + 1e-9
        means[c] = float(mean)
        stds[c] = float(std)
        full_df[c] = (full_df[c] - mean) / std
        full_df[c] = full_df[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return full_df, means, stds


# -----------------------------
# Split
# -----------------------------

def chrono_split(x_df: pd.DataFrame, y_df: pd.DataFrame, val_ratio: float = 0.2):
    ids = np.sort(x_df["anonymized_id"].unique())
    split_idx = int(len(ids) * (1 - val_ratio))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]
    x_train = x_df[x_df["anonymized_id"].isin(train_ids)].reset_index(drop=True)
    x_val = x_df[x_df["anonymized_id"].isin(val_ids)].reset_index(drop=True)
    y_train = y_df[y_df["anonymized_id"].isin(train_ids)].reset_index(drop=True)
    y_val = y_df[y_df["anonymized_id"].isin(val_ids)].reset_index(drop=True)
    return x_train, x_val, y_train, y_val, train_ids, val_ids


# -----------------------------
# Evaluation (implementation error)
# -----------------------------

def implementation_error_for_hour(
    positions: np.ndarray,
    hour_df: pd.DataFrame,
    volume_to_fill: float,
) -> float | None:
    close_series = hour_df["close"].dropna()
    if close_series.empty:
        return None
    close_price = float(close_series.iloc[-1])

    ask_prices = np.column_stack(
        [hour_df[f"ask_price_{lvl}"].to_numpy(dtype=float) for lvl in range(1, 6)]
    )
    bid_prices = np.column_stack(
        [hour_df[f"bid_price_{lvl}"].to_numpy(dtype=float) for lvl in range(1, 6)]
    )
    ask_vols = np.column_stack(
        [
            hour_df[f"ask_vol_{lvl}"].fillna(0.0).to_numpy(dtype=float)
            for lvl in range(1, 6)
        ]
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
        return None

    rel_err = abs(avg_price - close_price) / close_price
    fill_penalty = min(100.0, volume_to_fill / total_vol)
    return rel_err * fill_penalty * 10000.0


def evaluate_val(
    model: LOBSeq2Seq,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
    val_ids: np.ndarray,
    volume_to_fill: float,
    device: str,
) -> float:
    errors: List[float] = []
    model.eval()
    with torch.no_grad():
        for hour_id in val_ids:
            x_hour = x_val[x_val["anonymized_id"] == hour_id]
            y_hour = y_val[y_val["anonymized_id"] == hour_id]
            if len(x_hour) == 0 or len(y_hour) == 0:
                continue
            x_arr = x_hour[FEATURE_COLS].astype(np.float32).to_numpy()
            x_tensor = torch.from_numpy(x_arr).unsqueeze(0).to(device)
            mid_pred = model(x_tensor, y_teacher=None).cpu().numpy()[0]
            positions = schedule_from_mid_forecast(mid_pred, volume_to_fill)
            err = implementation_error_for_hour(positions, y_hour, volume_to_fill)
            if err is not None:
                errors.append(err)
    return float(np.mean(errors)) if errors else float("nan")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    x_df, y_df, volume_to_fill = load_btc_frames()
    x_df = enrich_x_columns(x_df)

    x_train, x_val, y_train, y_val, train_ids, val_ids = chrono_split(x_df, y_df)
    x_std, means, stds = standardize(x_train, pd.concat([x_train, x_val]), FEATURE_COLS)
    # Re-split standardized frame
    x_train = x_std[x_std["anonymized_id"].isin(train_ids)].reset_index(drop=True)
    x_val = x_std[x_std["anonymized_id"].isin(val_ids)].reset_index(drop=True)

    train_ds = LastMinuteSeq2SeqDataset(x_train, y_train)
    val_ds = LastMinuteSeq2SeqDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = LOBSeq2Seq(feature_dim=len(FEATURE_COLS)).to(args.device)
    cfg = TrainCfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, opt, cfg)
        val_loss = eval_epoch(model, val_loader, cfg)
        val_impl_err = evaluate_val(model, x_val, y_val, val_ids, volume_to_fill, cfg.device)
        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_impl_err_bps={val_impl_err:.4f}"
        )


if __name__ == "__main__":
    main()

