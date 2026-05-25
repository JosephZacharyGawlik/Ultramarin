from __future__ import annotations

import numpy as np
import pandas as pd

from execution_edge.data import ASK_VOL_COLS, BID_VOL_COLS, VISIBLE_LAST_SECOND, ensure_time_in_hour_timedelta


WINDOW_SECONDS = (60, 300, 900, 1800, 3540)


def _prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = ensure_time_in_hour_timedelta(frame).copy()
    prepared = (
        prepared.sort_values(["anonymized_id", "time_in_hour"])
        .drop_duplicates(subset=["anonymized_id", "time_in_hour"], keep="last")
        .reset_index(drop=True)
    )

    mid_price_1 = (prepared["ask_price_1"] + prepared["bid_price_1"]) / 2.0
    top_ask_vol_5 = prepared.loc[:, ASK_VOL_COLS].sum(axis=1, skipna=True)
    top_bid_vol_5 = prepared.loc[:, BID_VOL_COLS].sum(axis=1, skipna=True)
    top_total_vol_5 = top_ask_vol_5 + top_bid_vol_5

    prepared["mid_price_1"] = mid_price_1
    prepared["rel_spread_1"] = (prepared["ask_price_1"] - prepared["bid_price_1"]) / mid_price_1.replace(0.0, np.nan)
    prepared["top_ask_vol_5"] = top_ask_vol_5
    prepared["top_bid_vol_5"] = top_bid_vol_5
    prepared["top_vol_imbalance_5"] = (top_bid_vol_5 - top_ask_vol_5) / top_total_vol_5.replace(0.0, np.nan)
    prepared["microprice_1"] = (
        prepared["ask_price_1"] * prepared["bid_vol_1"]
        + prepared["bid_price_1"] * prepared["ask_vol_1"]
    ) / (prepared["ask_vol_1"] + prepared["bid_vol_1"]).replace(0.0, np.nan)
    prepared["trade_volume"] = prepared["volume"].fillna(0.0)
    prepared["close_missing"] = prepared["close"].isna().astype(float)
    prepared["volume_missing"] = prepared["volume"].isna().astype(float)

    log_mid = np.log(prepared["mid_price_1"])
    prepared["mid_log_return_1s"] = log_mid.groupby(prepared["anonymized_id"]).diff().fillna(0.0)
    prepared["abs_mid_log_return_1s"] = prepared["mid_log_return_1s"].abs()
    prepared["sq_mid_log_return_1s"] = prepared["mid_log_return_1s"] ** 2

    return prepared


def _build_last_snapshot_features(prepared: pd.DataFrame, volume_to_fill: float) -> pd.DataFrame:
    last_snapshot = (
        prepared.groupby("anonymized_id", sort=True)
        .tail(1)
        .loc[
            :,
            [
                "anonymized_id",
                "ask_price_1",
                "bid_price_1",
                "ask_vol_1",
                "bid_vol_1",
                "mid_price_1",
                "rel_spread_1",
                "microprice_1",
                "top_ask_vol_5",
                "top_bid_vol_5",
                "top_vol_imbalance_5",
                "trade_volume",
            ],
        ]
        .copy()
    )

    last_snapshot = last_snapshot.rename(
        columns={
            "ask_price_1": "last_ask_price_1",
            "bid_price_1": "last_bid_price_1",
            "ask_vol_1": "last_ask_vol_1",
            "bid_vol_1": "last_bid_vol_1",
            "mid_price_1": "last_mid_price_1",
            "rel_spread_1": "last_rel_spread_1",
            "microprice_1": "last_microprice_1",
            "top_ask_vol_5": "last_top_ask_vol_5",
            "top_bid_vol_5": "last_top_bid_vol_5",
            "top_vol_imbalance_5": "last_top_vol_imbalance_5",
            "trade_volume": "last_trade_volume",
        }
    )

    depth_columns = [
        "last_ask_vol_1",
        "last_bid_vol_1",
        "last_top_ask_vol_5",
        "last_top_bid_vol_5",
    ]
    for column in depth_columns:
        last_snapshot[f"{column}_to_fill"] = last_snapshot[column] / volume_to_fill

    return last_snapshot.sort_values("anonymized_id").reset_index(drop=True)


def _build_window_features(prepared: pd.DataFrame, window_seconds: int, volume_to_fill: float) -> pd.DataFrame:
    window_start = VISIBLE_LAST_SECOND - pd.to_timedelta(window_seconds - 1, unit="s")
    window_frame = prepared.loc[prepared["time_in_hour"] >= window_start]

    aggregated = (
        window_frame.groupby("anonymized_id", sort=True)
        .agg(
            row_count=("time_in_hour", "nunique"),
            rel_spread_mean=("rel_spread_1", "mean"),
            rel_spread_std=("rel_spread_1", "std"),
            ask_vol_1_mean=("ask_vol_1", "mean"),
            bid_vol_1_mean=("bid_vol_1", "mean"),
            top_ask_vol_5_mean=("top_ask_vol_5", "mean"),
            top_bid_vol_5_mean=("top_bid_vol_5", "mean"),
            top_vol_imbalance_5_mean=("top_vol_imbalance_5", "mean"),
            trade_volume_sum=("trade_volume", "sum"),
            trade_volume_mean=("trade_volume", "mean"),
            mid_return_mean=("mid_log_return_1s", "mean"),
            realized_vol=("sq_mid_log_return_1s", "sum"),
            abs_return_sum=("abs_mid_log_return_1s", "sum"),
            close_missing_rate=("close_missing", "mean"),
            volume_missing_rate=("volume_missing", "mean"),
        )
        .reset_index()
    )

    aggregated["realized_vol"] = np.sqrt(aggregated["realized_vol"])
    aggregated["coverage"] = aggregated["row_count"] / float(window_seconds)

    depth_columns = [
        "ask_vol_1_mean",
        "bid_vol_1_mean",
        "top_ask_vol_5_mean",
        "top_bid_vol_5_mean",
    ]
    for column in depth_columns:
        aggregated[f"{column}_to_fill"] = aggregated[column] / volume_to_fill

    aggregated = aggregated.rename(
        columns={column: f"w{window_seconds}_{column}" for column in aggregated.columns if column != "anonymized_id"}
    )
    aggregated[f"w{window_seconds}_rel_spread_std"] = aggregated[f"w{window_seconds}_rel_spread_std"].fillna(0.0)

    return aggregated


def build_hour_features_from_x(
    frame: pd.DataFrame,
    volume_to_fill: float,
    symbol: str,
    windows: tuple[int, ...] = WINDOW_SECONDS,
) -> pd.DataFrame:
    prepared = _prepare_feature_frame(frame)
    base = (
        prepared.groupby("anonymized_id", sort=True)
        ["time_in_hour"]
        .nunique()
        .rename("visible_rows")
        .reset_index()
        .sort_values("anonymized_id")
        .reset_index(drop=True)
    )
    base["symbol"] = symbol
    base["vol_to_fill"] = volume_to_fill

    feature_tables = [base, _build_last_snapshot_features(prepared, volume_to_fill)]
    for window_seconds in windows:
        feature_tables.append(_build_window_features(prepared, window_seconds, volume_to_fill))

    merged = feature_tables[0]
    for feature_table in feature_tables[1:]:
        merged = merged.merge(feature_table, on="anonymized_id", how="left")

    return merged.sort_values("anonymized_id").reset_index(drop=True)
