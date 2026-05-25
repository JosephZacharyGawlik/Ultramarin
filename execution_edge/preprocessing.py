"""Data preprocessing shared by the two main pipelines.

The deep-learning pipeline and the adaptive scheduling pipeline have
different preprocessing needs, so we expose two separate entry points.

``LOBProcessor`` is the deep-learning pipeline's cleaner. It cross-joins each
hour onto its full (anonymised_id, time_in_hour) grid, backfills leading
NaN prices, forward-fills remaining NaN prices, zero-fills volumes, and
returns a per-instrument z-score-normalised tensor ready to feed the
DeepLOB encoder.

``normalize_last_minute_frame`` is the adaptive scheduling pipeline's
last-minute reindexer. For each hour it reindexes the visible last-minute
frame onto the canonical 60-second grid (59:00 through 59:59 second by
second), leaving missing seconds as ``NaN`` rows. The walk-the-book
simulator handles those NaN rows gracefully at the affected second.
Both pipelines now share this last-minute handling so their holdout
evaluations operate on identical sets of hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import polars as pl
    import torch

from execution_edge.data import (
    ASK_PRICE_COLS,
    ASK_VOL_COLS,
    BID_PRICE_COLS,
    BID_VOL_COLS,
    LAST_MINUTE_INDEX,
    ensure_time_in_hour_timedelta,
)


# --------------------------------------------------------------------------- #
# Adaptive scheduling pipeline: last-minute reindexing                         #
# --------------------------------------------------------------------------- #

def normalize_last_minute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex each hour's last-minute frame to the canonical 60-second grid.

    Sorts and deduplicates, then for every ``anonymized_id`` reindexes the
    rows to ``LAST_MINUTE_INDEX`` (seconds 59:00 through 59:59 in
    ``Timedelta`` units), leaving missing seconds as ``NaN`` rows. Empty
    output for an hour means that hour has no observations in the
    last-minute window at all; ``build_hour_books`` further drops hours
    with no non-NaN ``close`` price.
    """
    normalized_frames: list[pd.DataFrame] = []
    prepared = (
        ensure_time_in_hour_timedelta(frame)
        .sort_values(["anonymized_id", "time_in_hour"])
        .drop_duplicates(subset=["anonymized_id", "time_in_hour"], keep="last")
    )

    for anonymized_id, hour_frame in prepared.groupby("anonymized_id", sort=True):
        hour_frame = hour_frame.drop(columns=["anonymized_id"], errors="ignore")
        reindexed = (
            hour_frame.sort_values("time_in_hour")
            .set_index("time_in_hour")
            .reindex(LAST_MINUTE_INDEX)
            .reset_index()
        )
        reindexed = reindexed.rename(columns={"index": "time_in_hour"})
        reindexed["anonymized_id"] = np.uint64(anonymized_id)
        normalized_frames.append(reindexed)

    if not normalized_frames:
        return prepared.iloc[0:0].copy()

    return pd.concat(normalized_frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Deep-learning pipeline: LOB cleaning + z-score normalisation                 #
# --------------------------------------------------------------------------- #

_PRICE_COLS_FOR_CLEANING = [
    "ask_price_1", "bid_price_1",
    "ask_price_2", "bid_price_2",
    "ask_price_3", "bid_price_3",
    "ask_price_4", "bid_price_4",
    "ask_price_5", "bid_price_5",
    "open", "high", "low", "close",
]

_VOL_COLS_FOR_CLEANING = [
    "ask_vol_1", "bid_vol_1",
    "ask_vol_2", "bid_vol_2",
    "ask_vol_3", "bid_vol_3",
    "ask_vol_4", "bid_vol_4",
    "ask_vol_5", "bid_vol_5",
    "volume",
]


@dataclass
class TrainConfig:
    """Hyperparameters for the deep-learning training driver.

    ``device`` defaults to CUDA if available, otherwise CPU. Construction
    is lazy: ``torch`` is only imported when this dataclass is instantiated.
    """
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    smooth_lambda: float = 0.01
    input_window: int = 600
    target_window: int = 60
    val_ratio: float = 0.2
    device: object = None  # populated to torch.device in __post_init__
    x_path: Optional[Path] = None
    y_path: Optional[Path] = None
    x_test_path: Optional[Path] = None
    feature_cols: Optional[list[str]] = None

    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _backfill_leading_nans(df, cols: list[str]):
    """For each column, fill leading NaNs with the first observed value."""
    import polars as pl
    for col in cols:
        first_valid_idx = df.select((pl.col(col).is_not_null()).arg_max()).item()
        if first_valid_idx == 0:
            continue
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) < first_valid_idx)
            .then(pl.lit(df[col][first_valid_idx]))
            .otherwise(pl.col(col))
            .alias(col)
        )
    return df


def _df_to_tensor(
    df,
    seq_len: int,
    id_col: str = "anonymized_id",
    time_col: str = "time_in_hour",
):
    """Pivot a long-format DataFrame to a tensor of shape (T, N_ids, F).

    Uses numpy for unique-ID enumeration to avoid the PyTorch uint64 limitation
    encountered on newer Colab images.
    """
    import polars as pl
    import torch
    df = df.with_columns([pl.col(time_col).cast(pl.Int32).alias(f"{time_col}_int")])
    time_int_col = f"{time_col}_int"
    feature_cols = [c for c in df.columns if c not in [id_col, time_col, time_int_col]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32).to(device)

    ids_np = df[id_col].to_numpy()
    times_np = df[time_int_col].to_numpy()
    times = torch.tensor(times_np).to(device)

    unique_ids_np, inverse = np.unique(ids_np, return_inverse=True)
    inverse_t = torch.tensor(inverse.astype(np.int64), dtype=torch.int64).to(device)
    num_ids = len(unique_ids_np)
    n_features = features.shape[1]
    X = torch.zeros(seq_len, num_ids, n_features, dtype=torch.float32, device=device)

    for i in range(num_ids):
        mask = (inverse_t == i)
        uid_features = features[mask]
        sorted_idx = torch.argsort(times[mask])
        X[:, i, :] = uid_features[sorted_idx]

    id_map = torch.tensor(
        np.stack([np.arange(num_ids, dtype=np.float64), unique_ids_np.astype(np.float64)], axis=1),
        dtype=torch.float64,
    ).to(device)
    return X, id_map


class LOBProcessor:
    """LOB cleaning and per-instrument z-score normalisation.

    Used by the deep-learning pipeline to turn raw ``X_train.parquet`` and
    ``y_train.parquet`` frames into normalised tensors. Cleaning steps:

    1. Sort and deduplicate by ``(anonymized_id, time_in_hour)``.
    2. Cross-join onto the full ``(id, time)`` grid so every hour has every
       second present.
    3. Backfill leading NaN price values per hour, then forward-fill the rest.
    4. Zero-fill missing volume values.
    5. Pivot to tensor and apply per-instrument z-score normalisation.
    """

    def __init__(self, config: TrainConfig, device=None):
        import torch
        self.config = config
        self.device = device or torch.device("cpu")
        self.feature_map: Optional[dict[str, int]] = None
        self.price_cols = _PRICE_COLS_FOR_CLEANING
        self.vol_cols = _VOL_COLS_FOR_CLEANING

    def _apply_cleaning(self, df):
        import polars as pl
        df = df.sort(["anonymized_id", "time_in_hour"])
        df = df.unique(subset=["anonymized_id", "time_in_hour"], keep="last")

        unique_times = df.select(pl.col("time_in_hour").unique().sort())
        unique_ids = df.select(pl.col("anonymized_id").unique())
        full_grid = unique_ids.join(unique_times, how="cross")
        df = full_grid.join(df, on=["anonymized_id", "time_in_hour"], how="left")
        df = df.sort(["anonymized_id", "time_in_hour"])

        price_cols_to_fill = list(self.price_cols)
        if "mid_price" in df.columns:
            price_cols_to_fill.append("mid_price")

        df = df.group_by("anonymized_id").map_groups(lambda g: _backfill_leading_nans(g, price_cols_to_fill))
        df = df.group_by("anonymized_id").map_groups(
            lambda g: g.with_columns(pl.col(price_cols_to_fill).fill_null(strategy="forward"))
        )
        df = df.with_columns(pl.col(self.vol_cols).fill_null(0))
        return df

    def process(self, X_df, y_df=None) -> dict:
        """Return normalised X (and optionally Y) tensors and the stats used."""
        X_clean = self._apply_cleaning(X_df)

        input_seq_len, target_seq_len = 3600 - 60, 60
        X_tensor, X_id_map = _df_to_tensor(X_clean, seq_len=input_seq_len)

        y_tensor: Optional[torch.Tensor] = None
        y_id_map: Optional[torch.Tensor] = None
        if y_df is not None:
            y_clean = self._apply_cleaning(y_df)
            y_tensor, y_id_map = _df_to_tensor(y_clean, seq_len=target_seq_len)

        X_tensor = X_tensor.to(self.device)
        if y_tensor is not None:
            y_tensor = y_tensor.to(self.device)

        if self.feature_map is None:
            exclude = ["anonymized_id", "time_in_hour"]
            feature_names = [c for c in X_clean.columns if c not in exclude]
            self.feature_map = {name: i for i, name in enumerate(feature_names)}

        means = X_tensor.mean(dim=0, keepdim=True)
        stds = X_tensor.std(dim=0, keepdim=True)
        stds[stds == 0] = 1.0

        X_normalized = (X_tensor - means) / stds
        y_normalized = (y_tensor - means) / stds if y_tensor is not None else None

        return {
            "X": X_normalized,
            "y": y_normalized,
            "means": means,
            "stds": stds,
            "X_id_map": X_id_map,
            "y_id_map": y_id_map,
            "feature_map": self.feature_map,
        }
