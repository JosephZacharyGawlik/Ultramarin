import string
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from pathlib import Path
import polars as pl

@dataclass
class TrainCfg:
    # Hyperparameters
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    smooth_lambda: float = 0.01
    loss: string = "mse"

    # Windows
    input_window: int = 300   # Look-back
    target_window: int = 60   # Horizon
    val_ratio: float = 0.2

    # Environment
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Features & Data
    x_path: Path = None
    y_path: Path = None
    x_test_path: Path = None
    feature_cols: list = None
    target: string = "mid"

class TensorTimeDataset(Dataset):
    def __init__(self, X, Y_full, Y_target, input_window: int):
        self.X = X
        self.Y = Y_full
        self.Y_target = Y_target
        self.input_window = input_window

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        x_sample = self.X[-self.input_window:, idx, :]
        y_full_sample = self.Y[:, idx, :]
        target = self.Y_target[idx, :]
        return x_sample, y_full_sample, target

class InferenceTensorDataset(TensorTimeDataset):
    def __init__(self, X, input_window: int):
        self.X = X
        self.input_window = input_window

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        x_sample = self.X[-self.input_window:, idx, :]
        return x_sample

class LOBProcessor:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device("cpu")
        self.feature_map = None

        self.price_cols = [
            'ask_price_1','bid_price_1','ask_price_2','bid_price_2',
            'ask_price_3','bid_price_3','ask_price_4','bid_price_4',
            'ask_price_5','bid_price_5','open','high','low','close'
        ]
        self.vol_cols = [
            'ask_vol_1','bid_vol_1','ask_vol_2','bid_vol_2',
            'ask_vol_3','bid_vol_3','ask_vol_4','bid_vol_4',
            'ask_vol_5','bid_vol_5','volume'
        ]

    def _apply_cleaning(self, df):
        # Sort and deduplicate
        df = df.sort(["anonymized_id", "time_in_hour"])
        df = df.unique(subset=["anonymized_id", "time_in_hour"], keep="last")

        # Cross-join grid to ensure all (id, time) pairs exist
        unique_times = df.select(pl.col("time_in_hour").unique().sort())
        unique_ids = df.select(pl.col("anonymized_id").unique())
        full_grid = unique_ids.join(unique_times, how="cross")
        df = full_grid.join(df, on=["anonymized_id", "time_in_hour"], how="left")

        # Sort to ensure time-continuity for filling
        df = df.sort(["anonymized_id", "time_in_hour"])

        price_cols_to_fill = self.price_cols.copy()
        if "target" in df.columns:
            price_cols_to_fill.append("target") # TODO calculate target after filling?

        # Backfill leading NaNs, then forward fill remaining prices
        from utils.utils import backfill_first_nans
        df = df.group_by("anonymized_id").map_groups(lambda g: backfill_first_nans(g, price_cols_to_fill))
        df = df.group_by("anonymized_id").map_groups(lambda g: g.with_columns(
            pl.col(price_cols_to_fill).fill_null(strategy="forward")
        ))
        # Volume fill 0
        df = df.with_columns(pl.col(self.vol_cols).fill_null(0))
        return df

    def process(self, X_df, y_df=None):
        X_clean = self._apply_cleaning(X_df)

        if y_df is not None:
            y_clean = self._apply_cleaning(y_df)

            input_seq_len, target_seq_len = 3600 - 60, 60

            from utils.utils import df_to_tensor
            X_tensor, X_id_map = df_to_tensor(X_clean, seq_len=input_seq_len)
            y_tensor, y_id_map = df_to_tensor(y_clean, seq_len=target_seq_len)
        else:
            from utils.utils import df_to_tensor

            input_seq_len = 3600 - 60
            X_tensor, X_id_map = df_to_tensor(X_clean, seq_len=input_seq_len)
            y_tensor, y_id_map = None, None

        X_tensor = X_tensor.to(self.device)
        if y_tensor is not None:
            y_tensor = y_tensor.to(self.device)

        if self.feature_map is None:
            exclude = ["anonymized_id", "time_in_hour"]
            feature_names = [col for col in X_clean.columns if col not in exclude]
            self.feature_map = {name: i for i, name in enumerate(feature_names)}

        # Fresh per-instrument stats for this call
        # Shape: [1, N_current, F]
        means = X_tensor.mean(dim=0, keepdim=True)
        stds = X_tensor.std(dim=0, keepdim=True)
        stds[stds == 0] = 1.0

        X_normalized = (X_tensor - means) / stds
        y_normalized = (y_tensor - means) / stds if y_tensor is not None else None

        result = {
            "X": X_normalized, "y": y_normalized,
            "means": means, "stds": stds,
            "X_id_map": X_id_map, "y_id_map": y_id_map,
            "feature_map": self.feature_map
        }

        return result
