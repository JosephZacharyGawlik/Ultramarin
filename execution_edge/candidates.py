from __future__ import annotations

import pandas as pd


DEFAULT_CANDIDATES = [
    ("last_rel_spread_1",),
    ("w60_trade_volume_sum",),
    ("w60_ask_vol_1_mean_to_fill",),
    ("w60_bid_vol_1_mean_to_fill",),
    ("last_rel_spread_1", "w60_trade_volume_sum"),
    ("last_rel_spread_1", "w60_ask_vol_1_mean_to_fill"),
    ("w60_trade_volume_sum", "w60_ask_vol_1_mean_to_fill"),
]


def candidate_feature_sets(train_features: pd.DataFrame, candidate_mode: str) -> list[tuple[str, ...]]:
    if candidate_mode == "default":
        return list(DEFAULT_CANDIDATES)

    if candidate_mode == "all_single":
        candidates = []
        for column in train_features.columns:
            if column in {"anonymized_id", "symbol"}:
                continue
            valid = train_features[column].dropna()
            if valid.nunique() <= 1:
                continue
            candidates.append((column,))
        return candidates

    raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")


def artifact_suffix(candidate_mode: str, bin_options: list[int]) -> str:
    if candidate_mode == "default" and bin_options == [5]:
        return ""
    return f"__{candidate_mode}_bins_{'_'.join(str(value) for value in bin_options)}"
