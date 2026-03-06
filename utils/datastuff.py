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

class TensorTimeDataset(Dataset):
    def __init__(self, X, Y_full, Y_mid, input_window: int):
        self.X = X      
        self.Y = Y_full 
        self.Y_mid = Y_mid 
        self.input_window = input_window

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        # Dynamically slice based on config input_window
        x_sample = self.X[-self.input_window:, idx, :] 
        y_full_sample = self.Y[:, idx, :]
        target = self.Y_mid[idx, :]
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
        self.means = None
        self.stds = None
        self.feature_map = None
        
        # Columns from your ref
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
        # 1. Sort
        df = df.sort(["anonymized_id", "time_in_hour"])

        # Arbitrarily pick the row that is later
        df = df.unique(subset=["anonymized_id", "time_in_hour"], keep="last")

        # ensure all ids are kept
        unique_times = df.select(pl.col("time_in_hour").unique().sort())
        unique_ids = df.select(pl.col("anonymized_id").unique())

        full_grid = unique_ids.join(unique_times, how="cross")

        df = full_grid.join(df, on=["anonymized_id", "time_in_hour"], how="left")
    
        # 5. Sort to ensure time-continuity for filling
        df = df.sort(["anonymized_id", "time_in_hour"])

        price_cols_to_fill = self.price_cols.copy()
        if "mid_price" in df.columns:
            price_cols_to_fill.append("mid_price")

        # 2. Backfill leading NaNs (Using your external helper)
        from utils.utils import backfill_first_nans # Ensure this import works
        df = df.group_by("anonymized_id").map_groups(lambda g: backfill_first_nans(g, price_cols_to_fill))
        # 3. Forward fill remaining prices
        df = df.group_by("anonymized_id").map_groups(lambda g: g.with_columns(
            pl.col(price_cols_to_fill).fill_null(strategy="forward")
        ))
        # 4. Volume fill 0
        df = df.with_columns(pl.col(self.vol_cols).fill_null(0))
        return df

    def process(self, X_df, y_df=None):
        X_clean = self._apply_cleaning(X_df)
        
        if y_df is not None:
            y_clean = self._apply_cleaning(y_df)

            seq_len_X, seq_len_y = 3600 - 60, 60
            
            # Convert to Tensors (Using your external helper)
            from utils.utils import df_to_tensor
            X_tens, X_id_map = df_to_tensor(X_clean, seq_len=seq_len_X)
            y_tens, y_id_map = df_to_tensor(y_clean, seq_len=seq_len_y)
        else:
            from utils.utils import df_to_tensor
            
            # --- THE FIX: Ensure exactly seq_len rows per ID ---
            seq_len_X = 3600 - 60
            
            X_tens, X_id_map = df_to_tensor(X_clean, seq_len=seq_len_X)
            y_tens, y_id_map = None, None

        X_tens = X_tens.to(self.device)
        if y_tens is not None: y_tens = y_tens.to(self.device)

        if self.feature_map is None:
            exclude = ["anonymized_id", "time_in_hour"]
            feature_names = [col for col in X_clean.columns if col not in exclude]
            self.feature_map = {name: i for i, name in enumerate(feature_names)}

        # --- 6. Per-instrument SCALING ---
        if self.means is None:
            # Training: compute mean/std per instrument across time
            # Shape: [1, num_ids, num_features]
            self.means = X_tens.mean(dim=0, keepdim=True)
            self.stds = X_tens.std(dim=0, keepdim=True)
            self.stds[self.stds == 0] = 1.0

        # Apply same normalization to both train and val (broadcasts automatically)
        X_norm = (X_tens - self.means) / self.stds
        y_norm = (y_tens - self.means) / self.stds if y_tens is not None else None

        result = {
            "X": X_norm, "y": y_norm,
            "means": self.means, "stds": self.stds,
            "X_id_map": X_id_map, "y_id_map": y_id_map,
            "feature_map": self.feature_map
        }

        if y_tens is not None and "mid_price" in self.feature_map:
            mid_idx = self.feature_map["mid_price"]

            # Extract raw midprice tensor: shape [seq_len, num_ids]
            mid_tensor = y_tens[:, :, mid_idx]

            # Compute per-instrument mean and std over time (dim=0)
            result["mid_mean"] = mid_tensor.mean(dim=0, keepdim=True)   # [1, num_ids]
            result["mid_stds"] = mid_tensor.std(dim=0, keepdim=True)    # [1, num_ids]
            result["mid_stds"][result["mid_stds"] == 0] = 1.0

        return result
