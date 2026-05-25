from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from execution_edge.preprocessing import normalize_last_minute_frame
from execution_edge.data import (
    ASK_PRICE_COLS,
    ASK_VOL_COLS,
    BID_PRICE_COLS,
    BID_VOL_COLS,
    DATA_DIR,
    LAST_MINUTE_INDEX,
    ensure_time_in_hour_timedelta,
)


@dataclass(frozen=True)
class HourBook:
    anonymized_id: int
    observed_rows: int
    ask_prices: np.ndarray
    ask_volumes: np.ndarray
    bid_prices: np.ndarray
    bid_volumes: np.ndarray
    close_price: float


@lru_cache(maxsize=1)
def load_simulator():
    simulator_path = Path(DATA_DIR) / "simulate_walk_the_book.py"
    spec = importlib.util.spec_from_file_location("execution_edge_simulator", simulator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import simulator from {simulator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate_walk_the_book




def build_hour_books(frame: pd.DataFrame) -> list[HourBook]:
    prepared = ensure_time_in_hour_timedelta(frame)
    normalized = normalize_last_minute_frame(prepared)
    books: list[HourBook] = []

    observed_rows = prepared.groupby("anonymized_id", sort=True)["time_in_hour"].nunique().to_dict()

    for anonymized_id, hour_frame in normalized.groupby("anonymized_id", sort=True):
        close_values = hour_frame["close"].dropna()
        if close_values.empty:
            continue

        books.append(
            HourBook(
                anonymized_id=int(anonymized_id),
                observed_rows=int(observed_rows.get(anonymized_id, 0)),
                ask_prices=hour_frame.loc[:, ASK_PRICE_COLS].to_numpy(dtype=float),
                ask_volumes=hour_frame.loc[:, ASK_VOL_COLS].to_numpy(dtype=float),
                bid_prices=hour_frame.loc[:, BID_PRICE_COLS].to_numpy(dtype=float),
                bid_volumes=hour_frame.loc[:, BID_VOL_COLS].to_numpy(dtype=float),
                close_price=float(close_values.iloc[-1]),
            )
        )

    return books


def score_schedule(
    schedule: np.ndarray,
    hour_book: HourBook,
    volume_to_fill: float,
) -> dict[str, float | int]:
    if len(schedule) != len(LAST_MINUTE_INDEX):
        raise ValueError("schedule must have exactly 60 elements")

    simulate_walk_the_book = load_simulator()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total_volume_executed, average_price = simulate_walk_the_book(
            schedule.astype(float, copy=False),
            hour_book.ask_prices,
            hour_book.ask_volumes,
            hour_book.bid_prices,
            hour_book.bid_volumes,
        )

    if total_volume_executed <= 0.0 or not np.isfinite(average_price):
        penalty = 100.0
        score_bps = penalty * 10000.0
    else:
        penalty = min(100.0, volume_to_fill / total_volume_executed)
        score_bps = (
            abs(float(average_price) - hour_book.close_price)
            / hour_book.close_price
            * penalty
            * 10000.0
        )

    fill_ratio = total_volume_executed / volume_to_fill if volume_to_fill else np.nan

    return {
        "anonymized_id": hour_book.anonymized_id,
        "observed_rows": hour_book.observed_rows,
        "close_price": hour_book.close_price,
        "total_volume_executed": float(total_volume_executed),
        "average_price": float(average_price) if np.isfinite(average_price) else np.nan,
        "fill_ratio": float(fill_ratio),
        "penalty": float(penalty),
        "score_bps": float(score_bps),
    }


def evaluate_schedule_family(
    schedules: Mapping[str, np.ndarray],
    hour_books: Iterable[HourBook],
    volume_to_fill: float,
) -> pd.DataFrame:
    books = list(hour_books)
    records: list[dict[str, float | int | str]] = []

    for schedule_id, schedule in schedules.items():
        for hour_book in books:
            record = score_schedule(schedule=schedule, hour_book=hour_book, volume_to_fill=volume_to_fill)
            record["schedule_id"] = schedule_id
            records.append(record)

    return pd.DataFrame(records)
