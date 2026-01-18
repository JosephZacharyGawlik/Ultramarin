#!/usr/bin/env python3
"""
Feature engineering and ridge regression baseline for last-minute execution targets.

The script:
1. Loads X/Y parquet files for a given symbol.
2. Engineers hour-level features from the first 59 minutes (X_train).
3. Builds targets from the final minute (y_train).
4. Splits hours into train/validation blocks.
5. Fits a multi-target ridge regressor implemented with NumPy.
6. Reports MAE/RMSE per target plus overall metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from simulate_walk_the_book import simulate_walk_the_book

# Time windows (seconds) used for contextual statistics
WINDOWS = {
    "last_60s": 60,
    "last_5m": 300,
    "last_15m": 900,
}

TARGET_COLUMNS = [
    "target_close",
    "target_mean_spread",
    "target_mean_bid_vol",
    "target_mean_ask_vol",
    "target_mean_top_depth",
]


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


def compute_implementation_error(
    submission_df: pd.DataFrame,
    y_df: pd.DataFrame,
    volume_to_fill: float,
) -> float:
    merged = submission_df.merge(
        y_df, on=["anonymized_id", "time_in_hour"], how="left", suffixes=("", "_y")
    ).sort_values(["anonymized_id", "time_in_hour"])

    errors: List[float] = []
    for hour_id, hour_df in merged.groupby("anonymized_id"):
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
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame


class MultiTargetRidge:
    """Closed-form ridge regression that solves for all targets simultaneously."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must be 2D arrays")

        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_aug = np.hstack([X, ones])

        xtx = X_aug.T @ X_aug
        xty = X_aug.T @ y

        reg = self.alpha * np.eye(xtx.shape[0], dtype=X.dtype)
        reg[-1, -1] = 0.0  # do not regularize bias term

        system = xtx + reg
        try:
            self.coef_ = np.linalg.solve(system, xty)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(system) @ xty

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model has not been fit yet.")
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_aug = np.hstack([X, ones])
        return X_aug @ self.coef_


def load_symbol_frames(root: Path, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = root / symbol
    x_path = base / "X_train.parquet"
    y_path = base / "y_train.parquet"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Could not find parquet files under {base}")

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

    df["log_mid"] = np.log(df["midprice"].clip(lower=1e-9))
    df["mid_return"] = (
        df.groupby("anonymized_id")["log_mid"].diff().fillna(0.0)
    )
    return df


def _window_slice(hour_df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    max_second = hour_df["second_of_hour"].max()
    cutoff = max_second - seconds
    return hour_df[hour_df["second_of_hour"] >= cutoff]


def _add_window_stats(
    features: Dict[str, float],
    hour_df: pd.DataFrame,
    window_name: str,
    window_seconds: int,
) -> None:
    window_df = _window_slice(hour_df, window_seconds)
    if window_df.empty:
        return
    prefix = f"feat_{window_name}"
    features[f"{prefix}_spread_mean"] = window_df["spread"].mean()
    features[f"{prefix}_spread_std"] = window_df["spread"].std(ddof=0)
    features[f"{prefix}_imb_mean"] = window_df["imbalance"].mean()
    features[f"{prefix}_depth_mean"] = window_df["total_depth"].mean()
    features[f"{prefix}_depth_ratio"] = (
        window_df["bid_depth_total"] / (window_df["ask_depth_total"] + 1e-9)
    ).mean()
    features[f"{prefix}_mid_trend"] = (
        window_df["midprice"].iloc[-1] - window_df["midprice"].iloc[0]
    )
    features[f"{prefix}_return_std"] = window_df["mid_return"].std(ddof=0)
    features[f"{prefix}_volume_sum"] = window_df["volume"].sum()


def hourly_feature_vector(hour_df: pd.DataFrame) -> pd.Series:
    feats: Dict[str, float] = {}
    feats["feat_num_rows"] = len(hour_df)
    feats["feat_mid_last"] = hour_df["midprice"].iloc[-1]
    feats["feat_mid_first"] = hour_df["midprice"].iloc[0]
    feats["feat_mid_change"] = feats["feat_mid_last"] - feats["feat_mid_first"]
    feats["feat_spread_mean"] = hour_df["spread"].mean()
    feats["feat_spread_std"] = hour_df["spread"].std(ddof=0)
    feats["feat_spread_last"] = hour_df["spread"].iloc[-1]
    feats["feat_imbalance_mean"] = hour_df["imbalance"].mean()
    feats["feat_imbalance_std"] = hour_df["imbalance"].std(ddof=0)
    feats["feat_imbalance_last"] = hour_df["imbalance"].iloc[-1]
    feats["feat_depth_ratio_mean"] = (
        hour_df["bid_depth_total"] / (hour_df["ask_depth_total"] + 1e-9)
    ).mean()
    feats["feat_total_depth_mean"] = hour_df["total_depth"].mean()
    feats["feat_total_depth_std"] = hour_df["total_depth"].std(ddof=0)
    feats["feat_volume_sum"] = hour_df["volume"].sum()
    feats["feat_volume_std"] = hour_df["volume"].std(ddof=0)
    feats["feat_volume_last"] = hour_df["volume"].iloc[-1]
    feats["feat_return_std"] = hour_df["mid_return"].std(ddof=0)
    feats["feat_return_mean"] = hour_df["mid_return"].mean()
    feats["feat_high_minus_low"] = hour_df["high"].max() - hour_df["low"].min()

    for window_name, seconds in WINDOWS.items():
        _add_window_stats(feats, hour_df, window_name, seconds)

    return pd.Series(feats)


def build_feature_matrix(x_df: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_x_columns(x_df)
    feature_rows: List[pd.Series] = []
    for hour_id, group in enriched.groupby("anonymized_id"):
        feats = hourly_feature_vector(group)
        feats["anonymized_id"] = hour_id
        feature_rows.append(feats)
    features = pd.DataFrame(feature_rows)
    return features


def build_targets(y_df: pd.DataFrame) -> pd.DataFrame:
    df = y_df.copy()
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["top_depth"] = df["ask_vol_1"] + df["bid_vol_1"]
    df["close"] = df.groupby("anonymized_id")["close"].ffill()
    df["close"] = df.groupby("anonymized_id")["close"].bfill()
    targets = (
        df.groupby("anonymized_id")
        .agg(
            target_close=("close", "last"),
            target_mean_spread=("spread", "mean"),
            target_mean_bid_vol=("bid_vol_1", "mean"),
            target_mean_ask_vol=("ask_vol_1", "mean"),
            target_mean_top_depth=("top_depth", "mean"),
        )
        .reset_index()
    )
    return targets


def assemble_dataset(
    x_df: pd.DataFrame, y_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = build_feature_matrix(x_df)
    targets = build_targets(y_df)
    dataset = features.merge(targets, on="anonymized_id", how="inner")
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    feature_cols = [c for c in dataset.columns if c.startswith("feat_")]
    target_cols = [c for c in TARGET_COLUMNS if c in dataset.columns]
    feature_frame = dataset[["anonymized_id", *feature_cols]]
    target_frame = dataset[["anonymized_id", *target_cols]]
    return feature_frame, target_frame


def train_val_split(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    val_fraction: float = 0.2,
) -> DatasetSplit:
    merged = features.merge(targets, on="anonymized_id", how="inner")
    merged = merged.sort_values("anonymized_id").reset_index(drop=True)

    feature_cols = [c for c in merged.columns if c.startswith("feat_")]
    target_cols = [c for c in TARGET_COLUMNS if c in merged.columns]

    all_ids = merged["anonymized_id"].unique()
    split_idx = max(1, int(len(all_ids) * (1 - val_fraction)))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    train_df = merged[merged["anonymized_id"].isin(train_ids)]
    val_df = merged[merged["anonymized_id"].isin(val_ids)]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df[target_cols].to_numpy(dtype=np.float64)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    y_val = val_df[target_cols].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train_scaled, X_val_scaled = scaler.fit_transform(X_train, X_val)

    return DatasetSplit(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        feature_names=feature_cols,
        target_names=target_cols,
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
    )


class StandardScaler:
    """Minimal feature scaler to keep the ridge system well-conditioned."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit_transform(
        self, X_train: np.ndarray, X_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.mean_ = X_train.mean(axis=0)
        self.scale_ = X_train.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self.transform(X_train), self.transform(X_val)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted.")
        return (X - self.mean_) / self.scale_


def read_volume_to_fill(data_root: Path, symbol: str) -> float:
    vol_path = data_root / symbol / "vol_to_fill.txt"
    if not vol_path.exists():
        raise FileNotFoundError(f"Missing {vol_path}")
    text = vol_path.read_text().strip()
    for token in text.replace(":", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    raise ValueError(f"Could not parse volume from {vol_path}")


def determine_start_second(pred_close: float, last_mid: float) -> int:
    if np.isnan(pred_close) or np.isnan(last_mid) or last_mid <= 0:
        return 20
    delta_ratio = (pred_close - last_mid) / last_mid
    if delta_ratio >= 0.001:
        return 0
    if delta_ratio >= 0.0002:
        return 10
    if delta_ratio >= -0.0002:
        return 20
    if delta_ratio >= -0.0008:
        return 35
    return 45


def simulate_hour_positions(
    hour_df: pd.DataFrame,
    pred_close: float,
    pred_mean_ask: float,
    start_second: int,
    volume_to_fill: float,
) -> List[float]:
    seconds = (
        (hour_df["time_in_hour"] - pd.Timedelta(minutes=59))
        .dt.total_seconds()
        .astype(int)
        .to_numpy()
    )
    ask_prices = hour_df["ask_price_1"].to_numpy()
    bid_prices = hour_df["bid_price_1"].to_numpy()
    ask_vols = hour_df["ask_vol_1"].fillna(0.0).to_numpy()

    window_len = max(1, 60 - max(0, start_second))
    remaining = float(volume_to_fill)
    positions: List[float] = []
    window_rows_seen = 0
    window_indices: List[int] = []

    if np.isnan(pred_mean_ask) or pred_mean_ask <= 1e-6:
        fallback = np.nanmean(ask_vols)
        norm_mean_ask = float(fallback) if fallback and fallback > 0 else 1.0
    else:
        norm_mean_ask = float(pred_mean_ask)

    for idx, sec in enumerate(seconds):
        if sec < start_second or remaining <= 0:
            positions.append(0.0)
            continue
        if window_rows_seen >= window_len:
            positions.append(0.0)
            continue

        remaining_slots = max(window_len - window_rows_seen, 1)
        base_allocation = remaining / remaining_slots

        ask_vol = ask_vols[idx]
        liquidity_ratio = ask_vol / norm_mean_ask if norm_mean_ask > 0 else 1.0
        liquidity_multiplier = float(np.clip(liquidity_ratio, 0.4, 1.6))

        ask_price = ask_prices[idx]
        bid_price = bid_prices[idx]
        if np.isnan(ask_price) or np.isnan(bid_price) or pred_close <= 0:
            price_multiplier = 1.0
        else:
            mid = (ask_price + bid_price) / 2.0
            gap_ratio = abs(mid - pred_close) / pred_close
            price_multiplier = float(np.clip(1.0 / (1.0 + gap_ratio * 150), 0.5, 1.2))

        multiplier = float(np.clip(liquidity_multiplier * price_multiplier, 0.4, 1.6))
        position = min(base_allocation * multiplier, remaining)

        positions.append(position)
        remaining -= position
        window_rows_seen += 1
        window_indices.append(len(positions) - 1)

    if remaining > 0 and window_indices:
        positions[window_indices[-1]] += remaining

    return positions


def generate_predicted_submission(
    y_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    volume_to_fill: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = {k: v for k, v in y_df.groupby("anonymized_id")}

    for _, pred_row in prediction_df.iterrows():
        hour_id = pred_row["anonymized_id"]
        hour_df = grouped.get(hour_id)
        if hour_df is None or hour_df.empty:
            continue
        pred_close = float(pred_row["target_close"])
        pred_mean_ask = float(pred_row.get("target_mean_ask_vol", np.nan))
        last_mid = float(pred_row.get("feat_mid_last", np.nan))
        start_second = determine_start_second(pred_close, last_mid)
        positions = simulate_hour_positions(
            hour_df, pred_close, pred_mean_ask, start_second, volume_to_fill
        )
        for time_value, position in zip(hour_df["time_in_hour"], positions):
            rows.append(
                {
                    "anonymized_id": hour_id,
                    "time_in_hour": time_value,
                    "position": float(position),
                }
            )

    return pd.DataFrame(rows)


def evaluate_predicted_strategy(
    model: MultiTargetRidge,
    split: DatasetSplit,
    y_df: pd.DataFrame,
    volume_to_fill: float,
) -> Tuple[float, float, float]:
    val_ids = split.val_df["anonymized_id"].to_numpy()
    predictions = model.predict(split.X_val)
    pred_df = pd.DataFrame(predictions, columns=split.target_names)
    pred_df["anonymized_id"] = val_ids
    pred_df = pred_df.merge(
        split.val_df[["anonymized_id", "feat_mid_last"]],
        on="anonymized_id",
        how="left",
    )

    y_val_df = y_df[y_df["anonymized_id"].isin(val_ids)].copy()
    submission_df = generate_predicted_submission(y_val_df, pred_df, volume_to_fill)

    if submission_df.empty:
        return float("nan"), float("nan"), float("nan")

    model_error = compute_implementation_error(
        submission_df, y_val_df, volume_to_fill
    )

    uniform_submission = baseline_uniform_submission(y_val_df, volume_to_fill)
    end_submission = baseline_end_of_hour_submission(
        y_val_df, volume_to_fill, seconds_from_end=20
    )

    baseline_uniform_error = compute_implementation_error(
        uniform_submission, y_val_df, volume_to_fill
    )
    baseline_end_error = compute_implementation_error(
        end_submission, y_val_df, volume_to_fill
    )

    return model_error, baseline_uniform_error, baseline_end_error


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
    n_features: int,
) -> Dict[str, Dict[str, float]]:
    residuals = y_true - y_pred
    metrics: Dict[str, Dict[str, float]] = {}
    n_samples = y_true.shape[0]
    adj_denom = max(n_samples - n_features - 1, 1)
    for idx, name in enumerate(target_names):
        errors = residuals[:, idx]
        ss_res = float(np.sum(errors**2))
        centered = y_true[:, idx] - np.mean(y_true[:, idx])
        ss_tot = float(np.sum(centered**2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        adj_r2 = 1.0 - (1.0 - r2) * (n_samples - 1) / adj_denom
        metrics[name] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "r2": r2,
            "adj_r2": adj_r2,
        }
    flat_errors = residuals.ravel()
    flat_centered = y_true.ravel() - np.mean(y_true.ravel())
    ss_res_flat = float(np.sum(flat_errors**2))
    ss_tot_flat = float(np.sum(flat_centered**2)) or 1.0
    r2_overall = 1.0 - ss_res_flat / ss_tot_flat
    adj_r2_overall = 1.0 - (1.0 - r2_overall) * (n_samples - 1) / adj_denom
    metrics["overall"] = {
        "mae": float(np.mean(np.abs(flat_errors))),
        "rmse": float(np.sqrt(np.mean(flat_errors**2))),
        "r2": r2_overall,
        "adj_r2": adj_r2_overall,
    }
    return metrics


def format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    lines = []
    for name, stats in metrics.items():
        lines.append(
            f"{name:20s} | MAE: {stats['mae']:.6f}, RMSE: {stats['rmse']:.6f}, R2: {stats['r2']:.4f}, AdjR2: {stats['adj_r2']:.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ridge regression baseline for execution targets."
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
        "--alpha",
        type=float,
        default=10.0,
        help="Ridge regularization strength.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_df, y_df = load_symbol_frames(args.data_root, args.symbol)
    feature_frame, target_frame = assemble_dataset(x_df, y_df)
    split = train_val_split(feature_frame, target_frame, args.val_fraction)

    model = MultiTargetRidge(alpha=args.alpha)
    model.fit(split.X_train, split.y_train)
    val_pred = model.predict(split.X_val)
    metrics = compute_metrics(
        split.y_val, val_pred, split.target_names, split.X_val.shape[1]
    )

    print(f"Symbol: {args.symbol}")
    print(f"Train hours: {split.X_train.shape[0]}, Validation hours: {split.X_val.shape[0]}")
    print(f"Feature count: {split.X_train.shape[1]}, Targets: {len(split.target_names)}")
    print("\nValidation metrics:")
    print(format_metrics(metrics))

    preview = list(zip(split.target_names, split.y_val[0], val_pred[0]))
    print("\nSample validation comparison (first hour):")
    for name, actual, pred in preview:
        print(f"{name:20s} | actual={actual:.6f}, predicted={pred:.6f}")

    volume_to_fill = read_volume_to_fill(args.data_root, args.symbol)
    model_error, uniform_error, end_error = evaluate_predicted_strategy(
        model, split, y_df, volume_to_fill
    )
    print("\nImplementation error (validation subset):")
    print(f"Ridge-guided strategy: {model_error:.2f} bps")
    print(f"Uniform baseline:    {uniform_error:.2f} bps")
    print(f"End-of-hour baseline:{end_error:.2f} bps")


if __name__ == "__main__":
    main()
