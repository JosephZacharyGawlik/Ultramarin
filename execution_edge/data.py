from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

ASK_PRICE_COLS = tuple(f"ask_price_{level}" for level in range(1, 6))
ASK_VOL_COLS = tuple(f"ask_vol_{level}" for level in range(1, 6))
BID_PRICE_COLS = tuple(f"bid_price_{level}" for level in range(1, 6))
BID_VOL_COLS = tuple(f"bid_vol_{level}" for level in range(1, 6))

LAST_MINUTE_START = pd.Timedelta(minutes=59)
LAST_MINUTE_INDEX = pd.timedelta_range(LAST_MINUTE_START, periods=60, freq="s")
VISIBLE_LAST_SECOND = pd.Timedelta(minutes=58, seconds=59)
FULL_VISIBLE_WINDOW_SECONDS = 59 * 60


def available_symbols(data_dir: Path = DATA_DIR) -> list[str]:
    symbols = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_dir():
            continue
        required = (
            path / "X_train.parquet",
            path / "y_train.parquet",
            path / "X_test.parquet",
            path / "vol_to_fill.txt",
        )
        if all(item.exists() for item in required):
            symbols.append(path.name)
    return symbols


def parse_symbols(symbols: Sequence[str] | None, data_dir: Path = DATA_DIR) -> list[str]:
    if not symbols:
        return available_symbols(data_dir)

    requested = [symbol.upper() for symbol in symbols]
    if "ALL" in requested:
        return available_symbols(data_dir)

    known = set(available_symbols(data_dir))
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"Unknown symbols: {', '.join(unknown)}")
    return sorted(requested)


def ensure_time_in_hour_timedelta(frame: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_timedelta64_dtype(frame["time_in_hour"]):
        return frame

    converted = frame.copy()
    converted["time_in_hour"] = pd.to_timedelta(converted["time_in_hour"])
    return converted


def load_parquet_split(symbol: str, split_name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / symbol / f"{split_name}.parquet"
    frame = pd.read_parquet(path)
    return ensure_time_in_hour_timedelta(frame)


def load_volume_to_fill(symbol: str, data_dir: Path = DATA_DIR) -> float:
    path = data_dir / symbol / "vol_to_fill.txt"
    match = re.search(r"([\d.]+)", path.read_text())
    if not match:
        raise ValueError(f"Unable to parse volume_to_fill from {path}")
    return float(match.group(1))


def artifact_path(*parts: str) -> Path:
    path = ARTIFACTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

