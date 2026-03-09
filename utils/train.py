import copy
import torch
import numpy as np
import pandas as pd
import polars as pl
from utils.datastuff import *
from models.Seq2SeqAttention import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.datastuff import TrainCfg, LOBProcessor
from utils.utils import compute_ofi

def chrono_split(x_df, y_df, val_ratio=0.2):
    ids = np.sort(x_df["anonymized_id"].unique())
    split = int(len(ids) * (1 - val_ratio))
    train_ids, val_ids = ids[:split], ids[split:]

    x_train = x_df[x_df["anonymized_id"].isin(train_ids)].reset_index(drop=True)
    x_val   = x_df[x_df["anonymized_id"].isin(val_ids)].reset_index(drop=True)
    y_train = y_df[y_df["anonymized_id"].isin(train_ids)].reset_index(drop=True)
    y_val   = y_df[y_df["anonymized_id"].isin(val_ids)].reset_index(drop=True)
    return x_train, x_val, y_train, y_val

def train_epoch(model, loader, optimizer, cfg: TrainCfg, tf_ratio: float = 1.0):
    model.train()
    total_loss, num_samples = 0.0, 0

    for x_batch, y_batch, target in loader:
        x_batch, target = x_batch.to(cfg.device), target.to(cfg.device)
        optimizer.zero_grad()

        pred = model(x_batch, y_teacher=target, tf_ratio=tf_ratio)
        loss = forecast_loss(pred, target, cfg.smooth_lambda, cfg.direction_lambda)

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
            pred = model(x_batch, y_teacher=None, tf_ratio=0.0)
            loss = forecast_loss(pred, target, cfg.smooth_lambda, cfg.direction_lambda)
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

    # Compute OFI features per instrument
    x_train_df = compute_ofi(x_train_df)
    x_val_df   = compute_ofi(x_val_df)
    y_train_df = compute_ofi(y_train_df)
    y_val_df   = compute_ofi(y_val_df)

    # 3. Preprocess with LOBProcessor
    processor = LOBProcessor(cfg, device=cfg.device)

    # Process Train (calculates means/stds)
    train_out = processor.process(pl.from_pandas(x_train_df), pl.from_pandas(y_train_df))
    X_train_tensor, Y_train_tensor = train_out["X"], train_out["y"]

    # Process Val (computes fresh per-instrument stats)
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
    # Extra features = everything after the first 20 LOB features (mid_price + OFI columns)
    num_extra_features = len(cfg.feature_cols) - 20
    model = SuperModel(
        horizon=cfg.target_window,
        dropout=cfg.dropout,
        ask_bid_idx=(ask_price_idx, bid_price_idx),
        num_extra_features=num_extra_features,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        # Scheduled teacher forcing: linear decay from 1.0 to tf_floor
        tf_ratio = max(cfg.tf_floor, 1.0 - epoch / max(cfg.epochs - 1, 1))

        train_loss = train_epoch(model, train_loader, optimizer, cfg, tf_ratio=tf_ratio)
        val_loss   = eval_epoch(model, val_loader, cfg)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | TF: {tf_ratio:.2f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {cfg.patience} epochs)")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val loss: {best_val_loss:.6f})")

    val_means = val_out["means"][:, :, feature_indices]
    val_stds  = val_out["stds"][:, :, feature_indices]

    scalers = {
        "feat_means": train_means,         # [1, N_train, F_reordered]
        "feat_stds": train_stds,           # [1, N_train, F_reordered]
        "val_feat_means": val_means,       # [1, N_val, F_reordered]
        "val_feat_stds": val_stds,         # [1, N_val, F_reordered]
    }
    return model, scalers, val_loader, val_id_map, processor

def forecast_loss(pred, target, smooth_lambda=0.02, direction_lambda=0.0):
    mse = (pred - target) ** 2
    loss = mse.mean()
    if smooth_lambda > 0:
        smooth = (pred[:, 1:] - pred[:, :-1]) ** 2
        loss = loss + smooth_lambda * smooth.mean()
    if direction_lambda > 0:
        dir_true = torch.sign(target[:, 1:] - target[:, :-1])
        dir_pred = pred[:, 1:] - pred[:, :-1]
        direction_loss = -torch.mean(dir_true * dir_pred)
        loss = loss + direction_lambda * direction_loss
    return loss
