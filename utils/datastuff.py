import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from pathlib import Path

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
    