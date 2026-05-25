"""Canonical SHA-1 hashed dev/holdout partition.

Pools ``anonymized_id`` values across all seven pairs, sorts them by SHA-1
hash, and takes the top fraction as the holdout. The partition is global:
every pair's holdout is the intersection of these holdout IDs with that pair's
own training data. The same partition is used by every approach in this
repository, so cross-pipeline comparisons are apples-to-apples.

Typical usage::

    from execution_edge.splits import compute_holdout_partition

    dev_ids, holdout_ids = compute_holdout_partition(
        data_root="data",
        symbols=("BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT",
                 "ADAUSDT", "DOGEUSDT", "XRPUSDT"),
        fraction=0.20,
    )
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "XRPUSDT",
)


def stable_hash(value: int) -> str:
    """SHA-1 hex digest of the decimal string representation of ``value``."""
    return hashlib.sha1(str(int(value)).encode("utf-8")).hexdigest()


def compute_holdout_partition(
    data_root: str | Path,
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
    fraction: float = 0.20,
) -> tuple[set[int], set[int]]:
    """Return ``(dev_ids, holdout_ids)`` as sets of ``anonymized_id`` values.

    Parameters
    ----------
    data_root : str or Path
        Directory containing one subfolder per symbol, each with an
        ``X_train.parquet`` file that has an ``anonymized_id`` column.
    symbols : iterable of str
        Symbols to pool over. Defaults to all seven challenge pairs.
    fraction : float
        Holdout fraction. ``0.20`` reproduces the report's split.

    Returns
    -------
    dev_ids, holdout_ids : tuple of set[int]
        Disjoint sets whose union is the full pool of ``anonymized_id``
        values present in the supplied symbols' training data.
    """
    data_root = Path(data_root)
    all_ids: set[int] = set()
    for sym in symbols:
        x_path = data_root / sym / "X_train.parquet"
        df = pd.read_parquet(x_path, columns=["anonymized_id"])
        all_ids.update(df["anonymized_id"].astype("uint64").unique().tolist())

    ordered = sorted(all_ids, key=lambda v: (stable_hash(v), int(v)))
    n_holdout = int(round(len(ordered) * fraction))
    holdout = set(int(v) for v in ordered[:n_holdout])
    dev = set(int(v) for v in ordered) - holdout
    return dev, holdout


def per_symbol_split(
    symbol_ids: Iterable[int],
    holdout_ids: set[int],
) -> tuple[list[int], list[int]]:
    """Split a single symbol's IDs into dev and holdout subsets.

    Convenience wrapper for the common case of taking the global holdout set
    and intersecting it with the IDs of one pair.
    """
    dev: list[int] = []
    held: list[int] = []
    for uid in symbol_ids:
        (held if int(uid) in holdout_ids else dev).append(int(uid))
    return dev, held
