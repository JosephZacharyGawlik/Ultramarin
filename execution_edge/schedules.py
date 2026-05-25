from __future__ import annotations

import numpy as np
import pandas as pd

from execution_edge.data import LAST_MINUTE_INDEX


def side_sign(side: str) -> float:
    normalized = side.lower()
    if normalized == "ask":
        return 1.0
    if normalized == "bid":
        return -1.0
    raise ValueError("side must be either 'ask' or 'bid'")


def schedule_label(k: int, alpha: float) -> str:
    return f"k={k}|alpha={alpha:g}"


def parse_schedule_id(schedule_id: str) -> tuple[int, float]:
    parts = {}
    for item in schedule_id.split("|"):
        key, value = item.split("=", 1)
        parts[key] = value
    return int(parts["k"]), float(parts["alpha"])


def alpha_grid_suffix(alphas: list[float]) -> str:
    if not alphas:
        raise ValueError("alphas must not be empty")
    if len(alphas) == 1 and float(alphas[0]) == 0.0:
        return ""

    def format_part(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return "__alphas_" + "_".join(format_part(float(value)) for value in alphas)


def build_twap_schedule(
    total_volume: float,
    k: int,
    side: str,
    alpha: float = 0.0,
    num_seconds: int = 60,
) -> np.ndarray:
    if not 1 <= k <= num_seconds:
        raise ValueError(f"k must be between 1 and {num_seconds}")

    weights = np.arange(1, k + 1, dtype=float)
    if alpha == 0.0:
        weights = np.ones(k, dtype=float)
    else:
        weights = weights**alpha

    schedule = np.zeros(num_seconds, dtype=float)
    signed_total_volume = side_sign(side) * total_volume
    schedule[-k:] = signed_total_volume * weights / weights.sum()
    return schedule


def cap_schedule(
    schedule: np.ndarray,
    max_abs_position: float,
    atol: float = 1e-12,
) -> np.ndarray:
    if max_abs_position <= 0.0:
        raise ValueError("max_abs_position must be positive")

    vector = np.asarray(schedule, dtype=float)
    signed_total = float(vector.sum())
    total_volume = abs(signed_total)
    if total_volume <= atol:
        return vector.copy()

    sign = np.sign(signed_total)
    if np.any(vector * sign < -atol):
        raise ValueError("cap_schedule expects a one-sided schedule")

    num_seconds = len(vector)
    if max_abs_position * num_seconds + atol < total_volume:
        raise ValueError("cap is infeasible for the requested total volume")

    magnitudes = np.abs(vector)
    lo = float(magnitudes.min() - max_abs_position)
    hi = float(magnitudes.max())

    def project(tau: float) -> np.ndarray:
        return np.clip(magnitudes - tau, 0.0, max_abs_position)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        projected_sum = float(project(mid).sum())
        if projected_sum > total_volume:
            lo = mid
        else:
            hi = mid

    capped = project(0.5 * (lo + hi))
    residual = total_volume - float(capped.sum())
    if abs(residual) > 1e-10:
        free = capped < (max_abs_position - atol)
        if free.any():
            capped[free] += residual / float(free.sum())
        else:
            capped *= total_volume / float(capped.sum())

    return sign * capped


def cap_schedule_preserve_support(
    schedule: np.ndarray,
    max_abs_position: float,
    atol: float = 1e-12,
) -> np.ndarray:
    vector = np.asarray(schedule, dtype=float)
    support = np.abs(vector) > atol
    if not support.any():
        return vector.copy()

    active = vector[support]
    active_count = int(support.sum())
    total_volume = float(np.abs(active).sum())
    if max_abs_position * active_count + atol < total_volume:
        raise ValueError("cap is infeasible on the active support")

    capped = vector.copy()
    capped[support] = cap_schedule(active, max_abs_position=max_abs_position, atol=atol)
    return capped


def build_submission_frame(
    anonymized_ids: pd.Series,
    k_predictions: pd.Series,
    total_volume: float,
    side: str,
    alpha: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for anonymized_id, k in zip(anonymized_ids.to_list(), k_predictions.to_list(), strict=True):
        schedule = build_twap_schedule(
            total_volume=total_volume,
            k=int(k),
            side=side,
            alpha=alpha,
            num_seconds=len(LAST_MINUTE_INDEX),
        )
        repeated_id = np.repeat(np.asarray([int(anonymized_id)], dtype="uint64"), len(LAST_MINUTE_INDEX))
        rows.append(
            pd.DataFrame(
                {
                    "anonymized_id": repeated_id,
                    "time_in_hour": LAST_MINUTE_INDEX,
                    "position": schedule,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def build_submission_frame_from_schedule_ids(
    anonymized_ids: pd.Series,
    schedule_predictions: pd.Series,
    total_volume: float,
    side: str,
) -> pd.DataFrame:
    rows = []
    for anonymized_id, schedule_id in zip(anonymized_ids.to_list(), schedule_predictions.to_list(), strict=True):
        k, alpha = parse_schedule_id(str(schedule_id))
        schedule = build_twap_schedule(
            total_volume=total_volume,
            k=k,
            side=side,
            alpha=alpha,
            num_seconds=len(LAST_MINUTE_INDEX),
        )
        repeated_id = np.repeat(np.asarray([int(anonymized_id)], dtype="uint64"), len(LAST_MINUTE_INDEX))
        rows.append(
            pd.DataFrame(
                {
                    "anonymized_id": repeated_id,
                    "time_in_hour": LAST_MINUTE_INDEX,
                    "position": schedule,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def build_submission_frame_from_schedule_matrix(
    anonymized_ids: pd.Series,
    schedule_matrix: np.ndarray,
) -> pd.DataFrame:
    if schedule_matrix.ndim != 2:
        raise ValueError("schedule_matrix must be 2-dimensional")
    if schedule_matrix.shape[0] != len(anonymized_ids):
        raise ValueError("schedule_matrix row count must match anonymized_ids length")
    if schedule_matrix.shape[1] != len(LAST_MINUTE_INDEX):
        raise ValueError("schedule_matrix must have exactly 60 columns")

    rows = []
    for anonymized_id, schedule in zip(anonymized_ids.to_list(), schedule_matrix, strict=True):
        repeated_id = np.repeat(np.asarray([int(anonymized_id)], dtype="uint64"), len(LAST_MINUTE_INDEX))
        rows.append(
            pd.DataFrame(
                {
                    "anonymized_id": repeated_id,
                    "time_in_hour": LAST_MINUTE_INDEX,
                    "position": np.asarray(schedule, dtype=float),
                }
            )
        )

    return pd.concat(rows, ignore_index=True)
