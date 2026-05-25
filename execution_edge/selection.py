from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


def sorted_id_folds(anonymized_ids: Sequence[int], n_splits: int) -> list[np.ndarray]:
    unique_ids = np.array(sorted(pd.unique(pd.Series(anonymized_ids))))
    return [fold for fold in np.array_split(unique_ids, n_splits) if len(fold)]


def cost_matrix_from_hourly_scores(
    hourly_scores: pd.DataFrame,
    action_column: str = "k",
    action_order: Sequence[object] | None = None,
) -> pd.DataFrame:
    wide = hourly_scores.pivot(index="anonymized_id", columns=action_column, values="score_bps").reset_index()
    wide.columns.name = None
    if action_order is not None:
        ordered_columns = ["anonymized_id", *list(action_order)]
        wide = wide.loc[:, ordered_columns]
    return wide.sort_values("anonymized_id").reset_index(drop=True)


def merge_features_and_costs(feature_frame: pd.DataFrame, cost_matrix: pd.DataFrame) -> pd.DataFrame:
    merged = feature_frame.merge(cost_matrix, on="anonymized_id", how="inner")
    return merged.sort_values("anonymized_id").reset_index(drop=True)


@dataclass(frozen=True)
class CandidateSpec:
    feature_names: tuple[str, ...]
    bins: int

    @property
    def label(self) -> str:
        return f"{'|'.join(self.feature_names)}@bins={self.bins}"


@dataclass
class QuantileBucketActionModel:
    feature_names: tuple[str, ...]
    bins: int
    action_values: tuple[object, ...]
    default_action: object | None = None
    bin_edges: dict[str, np.ndarray] = field(default_factory=dict)
    bucket_to_action: dict[tuple[int, ...], object] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "QuantileBucketActionModel":
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")

        for feature_name in self.feature_names:
            values = frame[feature_name].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                self.bin_edges[feature_name] = np.array([-0.5, 0.5], dtype=float)
                continue

            edges = np.quantile(values, np.linspace(0.0, 1.0, self.bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                center = float(values[0])
                edges = np.array([center - 0.5, center + 0.5], dtype=float)
            self.bin_edges[feature_name] = edges

        self.default_action = min(self.action_values, key=lambda action: frame[action].mean())

        bucket_frame = frame.copy()
        bucket_columns = []
        for feature_name in self.feature_names:
            column_name = f"__bin_{feature_name}"
            bucket_frame[column_name] = QuantileBucketModel._assign_bins(frame[feature_name], self.bin_edges[feature_name])
            bucket_columns.append(column_name)

        for bucket_key, bucket_rows in bucket_frame.groupby(bucket_columns, sort=True):
            if not isinstance(bucket_key, tuple):
                bucket_key = (int(bucket_key),)
            best_action = min(self.action_values, key=lambda action: bucket_rows[action].mean())
            self.bucket_to_action[tuple(int(value) for value in bucket_key)] = best_action

        return self

    def predict_action(self, frame: pd.DataFrame) -> pd.Series:
        if self.default_action is None:
            raise RuntimeError("Model must be fit before predict_action")

        bucket_codes = []
        for feature_name in self.feature_names:
            bucket_codes.append(QuantileBucketModel._assign_bins(frame[feature_name], self.bin_edges[feature_name]))

        predictions = []
        for bucket_key in zip(*bucket_codes, strict=True):
            predictions.append(self.bucket_to_action.get(tuple(int(value) for value in bucket_key), self.default_action))

        return pd.Series(predictions, index=frame.index, dtype=object)


@dataclass
class QuantileBucketModel:
    feature_names: tuple[str, ...]
    bins: int
    k_values: tuple[int, ...]
    default_k: int | None = None
    bin_edges: dict[str, np.ndarray] = field(default_factory=dict)
    bucket_to_k: dict[tuple[int, ...], int] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "QuantileBucketModel":
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")

        for feature_name in self.feature_names:
            values = frame[feature_name].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                self.bin_edges[feature_name] = np.array([-0.5, 0.5], dtype=float)
                continue

            edges = np.quantile(values, np.linspace(0.0, 1.0, self.bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                center = float(values[0])
                edges = np.array([center - 0.5, center + 0.5], dtype=float)
            self.bin_edges[feature_name] = edges

        self.default_k = min(self.k_values, key=lambda k: frame[k].mean())

        bucket_frame = frame.copy()
        bucket_columns = []
        for feature_name in self.feature_names:
            column_name = f"__bin_{feature_name}"
            bucket_frame[column_name] = self._assign_bins(frame[feature_name], self.bin_edges[feature_name])
            bucket_columns.append(column_name)

        for bucket_key, bucket_rows in bucket_frame.groupby(bucket_columns, sort=True):
            if not isinstance(bucket_key, tuple):
                bucket_key = (int(bucket_key),)
            best_k = min(self.k_values, key=lambda k: bucket_rows[k].mean())
            self.bucket_to_k[tuple(int(value) for value in bucket_key)] = int(best_k)

        return self

    def predict_k(self, frame: pd.DataFrame) -> pd.Series:
        if self.default_k is None:
            raise RuntimeError("Model must be fit before predict_k")

        bucket_codes = []
        for feature_name in self.feature_names:
            bucket_codes.append(self._assign_bins(frame[feature_name], self.bin_edges[feature_name]))

        predictions = []
        for bucket_key in zip(*bucket_codes, strict=True):
            predictions.append(self.bucket_to_k.get(tuple(int(value) for value in bucket_key), self.default_k))

        return pd.Series(predictions, index=frame.index, dtype=int)

    @staticmethod
    def _assign_bins(series: pd.Series, edges: np.ndarray) -> np.ndarray:
        codes = np.full(len(series), -1, dtype=int)
        mask = series.notna().to_numpy()
        if mask.any():
            values = series.to_numpy(dtype=float, copy=False)[mask]
            assigned = np.searchsorted(edges, values, side="right") - 1
            assigned = np.clip(assigned, 0, len(edges) - 2)
            codes[mask] = assigned.astype(int)
        return codes


def score_predicted_ks(
    frame: pd.DataFrame,
    predicted_ks: pd.Series,
    k_values: Sequence[int],
) -> np.ndarray:
    k_index = {int(k): index for index, k in enumerate(k_values)}
    costs = frame.loc[:, list(k_values)].to_numpy(dtype=float)
    row_indices = np.arange(len(costs))
    column_indices = np.array([k_index[int(k)] for k in predicted_ks.to_list()], dtype=int)
    return costs[row_indices, column_indices]


def score_predicted_actions(
    frame: pd.DataFrame,
    predicted_actions: pd.Series,
    action_values: Sequence[object],
) -> np.ndarray:
    action_index = {action: index for index, action in enumerate(action_values)}
    costs = frame.loc[:, list(action_values)].to_numpy(dtype=float)
    row_indices = np.arange(len(costs))
    column_indices = np.array([action_index[action] for action in predicted_actions.to_list()], dtype=int)
    return costs[row_indices, column_indices]


def cross_validate_global_best_k(
    merged: pd.DataFrame,
    k_values: Sequence[int],
    n_splits: int = 5,
) -> dict[str, object]:
    scores: list[float] = []
    chosen_ks: list[int] = []

    for test_ids in sorted_id_folds(merged["anonymized_id"], n_splits):
        train = merged.loc[~merged["anonymized_id"].isin(test_ids)]
        test = merged.loc[merged["anonymized_id"].isin(test_ids)]

        best_k = min(k_values, key=lambda k: train[k].mean())
        scores.extend(test[best_k].to_list())
        chosen_ks.extend([int(best_k)] * len(test))

    return {
        "mean_score_bps": float(np.mean(scores)),
        "chosen_k_counts": dict(sorted(Counter(chosen_ks).items())),
    }


def cross_validate_quantile_buckets(
    merged: pd.DataFrame,
    feature_names: Sequence[str],
    k_values: Sequence[int],
    bins: int = 5,
    n_splits: int = 5,
) -> dict[str, object]:
    scores: list[float] = []
    chosen_ks: list[int] = []

    feature_tuple = tuple(feature_names)
    for test_ids in sorted_id_folds(merged["anonymized_id"], n_splits):
        train = merged.loc[~merged["anonymized_id"].isin(test_ids)].reset_index(drop=True)
        test = merged.loc[merged["anonymized_id"].isin(test_ids)].reset_index(drop=True)

        model = QuantileBucketModel(
            feature_names=feature_tuple,
            bins=bins,
            k_values=tuple(sorted(int(k) for k in k_values)),
        ).fit(train)
        predicted_ks = model.predict_k(test)
        scores.extend(score_predicted_ks(test, predicted_ks, k_values).tolist())
        chosen_ks.extend(predicted_ks.astype(int).to_list())

    return {
        "mean_score_bps": float(np.mean(scores)),
        "chosen_k_counts": dict(sorted(Counter(chosen_ks).items())),
    }


def cross_validate_global_best_action(
    merged: pd.DataFrame,
    action_values: Sequence[object],
    n_splits: int = 5,
) -> dict[str, object]:
    scores: list[float] = []
    chosen_actions: list[object] = []

    for test_ids in sorted_id_folds(merged["anonymized_id"], n_splits):
        train = merged.loc[~merged["anonymized_id"].isin(test_ids)]
        test = merged.loc[merged["anonymized_id"].isin(test_ids)]

        best_action = min(action_values, key=lambda action: train[action].mean())
        predicted_actions = pd.Series([best_action] * len(test), index=test.index, dtype=object)
        scores.extend(score_predicted_actions(test, predicted_actions, action_values).tolist())
        chosen_actions.extend([best_action] * len(test))

    return {
        "mean_score_bps": float(np.mean(scores)),
        "chosen_action_counts": dict(sorted(Counter(chosen_actions).items(), key=lambda item: str(item[0]))),
    }


def cross_validate_quantile_action_buckets(
    merged: pd.DataFrame,
    feature_names: Sequence[str],
    action_values: Sequence[object],
    bins: int = 5,
    n_splits: int = 5,
) -> dict[str, object]:
    scores: list[float] = []
    chosen_actions: list[object] = []

    feature_tuple = tuple(feature_names)
    for test_ids in sorted_id_folds(merged["anonymized_id"], n_splits):
        train = merged.loc[~merged["anonymized_id"].isin(test_ids)].reset_index(drop=True)
        test = merged.loc[merged["anonymized_id"].isin(test_ids)].reset_index(drop=True)

        model = QuantileBucketActionModel(
            feature_names=feature_tuple,
            bins=bins,
            action_values=tuple(action_values),
        ).fit(train)
        predicted_actions = model.predict_action(test)
        scores.extend(score_predicted_actions(test, predicted_actions, action_values).tolist())
        chosen_actions.extend(predicted_actions.to_list())

    return {
        "mean_score_bps": float(np.mean(scores)),
        "chosen_action_counts": dict(sorted(Counter(chosen_actions).items(), key=lambda item: str(item[0]))),
    }


def nested_cross_validate_quantile_search(
    merged: pd.DataFrame,
    candidate_specs: Sequence[CandidateSpec],
    k_values: Sequence[int],
    outer_splits: int = 5,
    inner_splits: int = 4,
) -> dict[str, object]:
    outer_scores: list[float] = []
    predicted_ks: list[int] = []
    selected_specs: list[str] = []
    fold_rows: list[dict[str, object]] = []

    outer_folds = sorted_id_folds(merged["anonymized_id"], outer_splits)
    for fold_index, outer_test_ids in enumerate(outer_folds, start=1):
        outer_train = merged.loc[~merged["anonymized_id"].isin(outer_test_ids)].reset_index(drop=True)
        outer_test = merged.loc[merged["anonymized_id"].isin(outer_test_ids)].reset_index(drop=True)

        inner_fold_count = min(inner_splits, len(pd.unique(outer_train["anonymized_id"])))
        if inner_fold_count < 2:
            raise ValueError("Nested CV requires at least 2 inner folds")

        best_spec: CandidateSpec | None = None
        best_inner_score: float | None = None

        for candidate_spec in candidate_specs:
            result = cross_validate_quantile_buckets(
                merged=outer_train,
                feature_names=candidate_spec.feature_names,
                k_values=k_values,
                bins=candidate_spec.bins,
                n_splits=inner_fold_count,
            )
            score = float(result["mean_score_bps"])
            if best_spec is None or score < best_inner_score or (
                score == best_inner_score and candidate_spec.label < best_spec.label
            ):
                best_spec = candidate_spec
                best_inner_score = score

        assert best_spec is not None
        assert best_inner_score is not None

        final_model = QuantileBucketModel(
            feature_names=best_spec.feature_names,
            bins=best_spec.bins,
            k_values=tuple(sorted(int(k) for k in k_values)),
        ).fit(outer_train)
        outer_predicted_ks = final_model.predict_k(outer_test)
        outer_costs = score_predicted_ks(outer_test, outer_predicted_ks, k_values)

        outer_scores.extend(outer_costs.tolist())
        predicted_ks.extend(outer_predicted_ks.astype(int).to_list())
        selected_specs.append(best_spec.label)
        fold_rows.append(
            {
                "outer_fold": fold_index,
                "outer_test_hours": len(outer_test),
                "selected_feature_names": "|".join(best_spec.feature_names),
                "selected_bins": best_spec.bins,
                "inner_cv_score_bps": best_inner_score,
                "outer_fold_score_bps": float(np.mean(outer_costs)),
                "predicted_k_counts": dict(sorted(Counter(outer_predicted_ks.astype(int).to_list()).items())),
            }
        )

    return {
        "mean_score_bps": float(np.mean(outer_scores)),
        "selected_spec_counts": dict(sorted(Counter(selected_specs).items())),
        "predicted_k_counts": dict(sorted(Counter(predicted_ks).items())),
        "fold_rows": fold_rows,
    }


def nested_cross_validate_quantile_action_search(
    merged: pd.DataFrame,
    candidate_specs: Sequence[CandidateSpec],
    action_values: Sequence[object],
    outer_splits: int = 5,
    inner_splits: int = 4,
) -> dict[str, object]:
    outer_scores: list[float] = []
    predicted_actions: list[object] = []
    selected_specs: list[str] = []
    fold_rows: list[dict[str, object]] = []

    outer_folds = sorted_id_folds(merged["anonymized_id"], outer_splits)
    for fold_index, outer_test_ids in enumerate(outer_folds, start=1):
        outer_train = merged.loc[~merged["anonymized_id"].isin(outer_test_ids)].reset_index(drop=True)
        outer_test = merged.loc[merged["anonymized_id"].isin(outer_test_ids)].reset_index(drop=True)

        inner_fold_count = min(inner_splits, len(pd.unique(outer_train["anonymized_id"])))
        if inner_fold_count < 2:
            raise ValueError("Nested CV requires at least 2 inner folds")

        best_spec: CandidateSpec | None = None
        best_inner_score: float | None = None

        for candidate_spec in candidate_specs:
            result = cross_validate_quantile_action_buckets(
                merged=outer_train,
                feature_names=candidate_spec.feature_names,
                action_values=action_values,
                bins=candidate_spec.bins,
                n_splits=inner_fold_count,
            )
            score = float(result["mean_score_bps"])
            if best_spec is None or score < best_inner_score or (
                score == best_inner_score and candidate_spec.label < best_spec.label
            ):
                best_spec = candidate_spec
                best_inner_score = score

        assert best_spec is not None
        assert best_inner_score is not None

        final_model = QuantileBucketActionModel(
            feature_names=best_spec.feature_names,
            bins=best_spec.bins,
            action_values=tuple(action_values),
        ).fit(outer_train)
        outer_predicted_actions = final_model.predict_action(outer_test)
        outer_costs = score_predicted_actions(outer_test, outer_predicted_actions, action_values)

        outer_scores.extend(outer_costs.tolist())
        predicted_actions.extend(outer_predicted_actions.to_list())
        selected_specs.append(best_spec.label)
        fold_rows.append(
            {
                "outer_fold": fold_index,
                "outer_test_hours": len(outer_test),
                "selected_feature_names": "|".join(best_spec.feature_names),
                "selected_bins": best_spec.bins,
                "inner_cv_score_bps": best_inner_score,
                "outer_fold_score_bps": float(np.mean(outer_costs)),
                "predicted_action_counts": dict(sorted(Counter(outer_predicted_actions.to_list()).items(), key=lambda item: str(item[0]))),
            }
        )

    return {
        "mean_score_bps": float(np.mean(outer_scores)),
        "selected_spec_counts": dict(sorted(Counter(selected_specs).items())),
        "predicted_action_counts": dict(sorted(Counter(predicted_actions).items(), key=lambda item: str(item[0]))),
        "fold_rows": fold_rows,
    }
