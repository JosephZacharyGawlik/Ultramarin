import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from pathlib import Path
import polars as pl

@dataclass
class TrainCfg:
    epochs: int = 15
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    smooth_lambda: float = 0.02
    val_ratio: float = 0.2
    input_window: int = 300   # Look-back period
    target_window: int = 60   # Prediction horizon

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
    
class LOBProcessor:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device("cpu")
        self.means = None
        self.stds = None
        self.feature_map = None
        self.id_to_idx = None  # Crucial for matching Val IDs to Train Stats
        
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
        # 2. Backfill leading NaNs (Using your external helper)
        from utils.utils import backfill_first_nans # Ensure this import works
        df = df.group_by("anonymized_id").map_groups(lambda g: backfill_first_nans(g, self.price_cols))
        # 3. Forward fill remaining prices
        df = df.group_by("anonymized_id").map_groups(lambda g: g.with_columns(
            pl.col(self.price_cols).fill_null(strategy="forward")
        ))
        # 4. Volume fill 0
        df = df.with_columns(pl.col(self.vol_cols).fill_null(0))
        return df

    def process(self, X_df, y_df=None):
        X_clean = self._apply_cleaning(X_df)
        
        if y_df is not None:
            y_clean = self._apply_cleaning(y_df)
            
            # --- 5. STRICT FILTERING (Duplicates & Seq Length) ---
            dup_ids = (
                X_clean.group_by(["anonymized_id", "time_in_hour"])
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") > 1)
                .select("anonymized_id").unique()
            )["anonymized_id"].to_list()

            seq_len_X, seq_len_y = 3600 - 60, 60
            valid_ids_X = X_clean.group_by("anonymized_id").agg(pl.len().alias("n")).filter(pl.col("n") == seq_len_X)
            valid_ids_y = y_clean.group_by("anonymized_id").agg(pl.len().alias("n")).filter(pl.col("n") == seq_len_y)
            
            valid_ids = valid_ids_X.join(valid_ids_y, on="anonymized_id", how="inner")["anonymized_id"].to_list()

            X_clean = X_clean.filter(~pl.col("anonymized_id").is_in(dup_ids)).filter(pl.col("anonymized_id").is_in(valid_ids))
            y_clean = y_clean.filter(~pl.col("anonymized_id").is_in(dup_ids)).filter(pl.col("anonymized_id").is_in(valid_ids))
            
            # Convert to Tensors (Using your external helper)
            from utils.utils import df_to_tensor
            X_tens, X_id_map = df_to_tensor(X_clean, seq_len=seq_len_X)
            y_tens, y_id_map = df_to_tensor(y_clean, seq_len=seq_len_y)
        else:
            from utils.utils import df_to_tensor
            X_tens, X_id_map = df_to_tensor(X_clean, seq_len=3600-60)
            y_tens, y_id_map = None, None

        X_tens = X_tens.to(self.device)
        if y_tens is not None: y_tens = y_tens.to(self.device)

        if self.feature_map is None:
            exclude = ["anonymized_id", "time_in_hour"]
            feature_names = [col for col in X_clean.columns if col not in exclude]
            self.feature_map = {name: i for i, name in enumerate(feature_names)}

        # --- 6. ID-BASED SCALING ---
        if self.means is None:
            # Training logic: Calculate and store mapping
            self.means = X_tens.mean(dim=0, keepdim=True)
            self.stds = X_tens.std(dim=0, keepdim=True)
            self.stds[self.stds == 0] = 1.0
            
            # id_to_idx maps the actual ID value to its column position in the mean tensor
            self.id_to_idx = {int(uid): i for i, uid in enumerate(X_id_map[:, 1].tolist())}
            cur_means, cur_stds = self.means, self.stds
        else:
            # Validation logic: Select correct means for current IDs
            current_ids = X_id_map[:, 1].tolist()
            idx_selector = [self.id_to_idx.get(int(uid), 0) for uid in current_ids]
            idx_tensor = torch.tensor(idx_selector, device=self.device)
            
            cur_means = self.means[:, idx_tensor, :]
            cur_stds = self.stds[:, idx_tensor, :]

        X_norm = (X_tens - cur_means) / cur_stds
        y_norm = (y_tens - cur_means) / cur_stds if y_tens is not None else None

        return {
            "X": X_norm, "y": y_norm, 
            "means": self.means, "stds": self.stds,
            "X_id_map": X_id_map, "y_id_map": y_id_map,
            "feature_map": self.feature_map
        }
