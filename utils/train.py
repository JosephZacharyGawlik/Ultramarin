import torch
import numpy as np
import pandas as pd
import polars as pl
from utils.datastuff import *
from models.Seq2SeqAttention import *
from torch.utils.data import DataLoader
from utils.datastuff import TrainCfg, LOBProcessor

def chrono_split(x_df, y_df, val_ratio=0.2):
    ids = np.sort(x_df["anonymized_id"].unique())
    split = int(len(ids) * (1 - val_ratio))
    train_ids, val_ids = ids[:split], ids[split:]

    x_train = x_df[x_df["anonymized_id"].isin(train_ids)].reset_index(drop=True)
    x_val   = x_df[x_df["anonymized_id"].isin(val_ids)].reset_index(drop=True)
    y_train = y_df[y_df["anonymized_id"].isin(train_ids)].reset_index(drop=True)
    y_val   = y_df[y_df["anonymized_id"].isin(val_ids)].reset_index(drop=True)
    return x_train, x_val, y_train, y_val

def train_epoch(model, loader, optimizer, cfg: TrainCfg):
    model.train()
    total_loss, num_samples = 0.0, 0

    for x_batch, y_batch, target in loader:
        x_batch, target = x_batch.to(cfg.device), target.to(cfg.device)
        optimizer.zero_grad()

        pred = model(x_batch, y_teacher=target)
        loss = forecast_loss(pred, target, cfg.smooth_lambda)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x_batch.size(0)
        num_samples += x_batch.size(0)
    return total_loss / max(num_samples, 1)

def eval_epoch(model, loader, cfg: TrainCfg):
    model.eval()
    total_loss, num_samples = 0.0, 0
    with torch.no_grad():
        for x_batch, _, target in loader:
            x_batch, target = x_batch.to(cfg.device), target.to(cfg.device)
            # No teacher forcing during evaluation
            pred = model(x_batch, y_teacher=None)
            loss = forecast_loss(pred, target, cfg.smooth_lambda)
            total_loss += float(loss.item()) * x_batch.size(0)
            num_samples += x_batch.size(0)
    return total_loss / max(num_samples, 1)

def train_val(cfg: TrainCfg = TrainCfg()):

    # 1. Load Data
    X_raw = pd.read_parquet(cfg.x_path).sort_values(["anonymized_id", "time_in_hour"])
    Y_raw = pd.read_parquet(cfg.y_path).sort_values(["anonymized_id", "time_in_hour"])

    # 2. Split
    x_train_df, x_val_df, y_train_df, y_val_df = chrono_split(X_raw, Y_raw, val_ratio=cfg.val_ratio)

    # Compute raw mid_price and add to ALL DataFrames before normalization.
    # X and Y must have the same features so normalization broadcasts correctly.
    x_train_df = x_train_df.assign(mid_price=(x_train_df["ask_price_1"] + x_train_df["bid_price_1"]) / 2.0)
    x_val_df   = x_val_df.assign(mid_price=(x_val_df["ask_price_1"]     + x_val_df["bid_price_1"])   / 2.0)
    y_train_df = y_train_df.assign(mid_price=(y_train_df["ask_price_1"] + y_train_df["bid_price_1"]) / 2.0)
    y_val_df   = y_val_df.assign(mid_price=(y_val_df["ask_price_1"]     + y_val_df["bid_price_1"])   / 2.0)

    # 3. Preprocess with LOBProcessor
    processor = LOBProcessor(cfg, device=cfg.device)

    # Process Train (calculates means/stds)
    train_out = processor.process(pl.from_pandas(x_train_df), pl.from_pandas(y_train_df))
    X_train_tensor, Y_train_tensor = train_out["X"], train_out["y"]

    # Process Val (reuses training means/stds)
    val_out = processor.process(pl.from_pandas(x_val_df), pl.from_pandas(y_val_df))
    X_val_tensor, Y_val_tensor, val_id_map = val_out["X"], val_out["y"], val_out["y_id_map"]

    # 4. Enforce FEATURE_COLS order and extract scalers
    feature_indices = [processor.feature_map[col] for col in cfg.feature_cols]

    X_train_tensor = X_train_tensor[:, :, feature_indices]
    X_val_tensor   = X_val_tensor[:, :, feature_indices]
    train_means    = train_out["means"][:, :, feature_indices]
    train_stds     = train_out["stds"][:, :, feature_indices]

    # Map feature names to their index in the reordered tensor
    feature_index_map = {col: i for i, col in enumerate(cfg.feature_cols)}
    ask_price_idx = feature_index_map["ask_price_1"]
    bid_price_idx = feature_index_map["bid_price_1"]

    # 5. Extract normalized mid_price for training target
    mid_price_idx  = feature_index_map["mid_price"]
    Y_train_tensor = Y_train_tensor[:, :, feature_indices]
    Y_val_tensor   = Y_val_tensor[:, :, feature_indices]

    mid_train = Y_train_tensor[:, :, mid_price_idx]  # [60, num_ids]
    mid_val   = Y_val_tensor[:, :, mid_price_idx]    # [60, num_ids]

    # mid_train shape [60, Num_IDs] -> .T makes [Num_IDs, 60]
    train_dataset = TensorTimeDataset(X_train_tensor, Y_train_tensor, mid_train.T, input_window=cfg.input_window)
    val_dataset   = TensorTimeDataset(X_val_tensor, Y_val_tensor, mid_val.T, input_window=cfg.input_window)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 6. Model & Training
    model = SuperModel(
        horizon=cfg.target_window,
        ask_bid_idx=(ask_price_idx, bid_price_idx)
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, cfg)
        val_loss   = eval_epoch(model, val_loader, cfg)
        print(f"Epoch {epoch+1:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    scalers = {
        "feat_means": train_means,
        "feat_stds": train_stds,
        "mid_mean": train_out["mid_mean"],       # per-instrument [1, num_train_ids], raw space
        "mid_stds": train_out["mid_stds"],       # per-instrument [1, num_train_ids], raw space
        "val_mid_mean": val_out["mid_mean"],     # per-instrument [1, num_val_ids], raw space
        "val_mid_stds": val_out["mid_stds"],     # per-instrument [1, num_val_ids], raw space
    }
    return model, scalers, val_loader, val_id_map, processor

def forecast_loss(pred, target, smooth_lambda=0.02):
    mse = (pred - target) ** 2
    loss = mse.mean()
    if smooth_lambda > 0:
        smooth = (pred[:, 1:] - pred[:, :-1]) ** 2
        loss = loss + smooth_lambda * smooth.mean()
    return loss
