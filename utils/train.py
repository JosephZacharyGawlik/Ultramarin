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
    tr_ids, va_ids = ids[:split], ids[split:]
    
    x_tr = x_df[x_df["anonymized_id"].isin(tr_ids)].reset_index(drop=True)
    x_va = x_df[x_df["anonymized_id"].isin(va_ids)].reset_index(drop=True)
    y_tr = y_df[y_df["anonymized_id"].isin(tr_ids)].reset_index(drop=True)
    y_va = y_df[y_df["anonymized_id"].isin(va_ids)].reset_index(drop=True)
    return x_tr, x_va, y_tr, y_va

def train_epoch(model, loader, opt, cfg: TrainCfg):
    model.train()
    total_loss, n = 0.0, 0
    
    # Unpack the 3 values yielded by TensorTimeDataset
    # xb: [Batch, 3540, Features]
    # yb: [Batch, 60, Features]
    # target: [Batch, 60] <--- This is the one you want for loss
    for xb, yb, target in loader:
        xb, target = xb.to(cfg.device), target.to(cfg.device)
        opt.zero_grad()
        
        # pred will be [Batch, 60]
        pred = model(xb, y_teacher=target)
        
        # Now both are length 60
        loss = forecast_loss(pred, target, cfg.smooth_lambda)
        
        loss.backward()
        opt.step()
        
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)

def eval_epoch(model, loader, cfg: TrainCfg):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, _, target in loader:
            xb, target = xb.to(cfg.device), target.to(cfg.device)
            # No teacher forcing during evaluation
            pred = model(xb, y_teacher=None)
            loss = forecast_loss(pred, target, cfg.smooth_lambda)
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total_loss / max(n, 1)

def train_val(cfg: TrainCfg = TrainCfg()):

    # 1. Load Data
    X_raw = pd.read_parquet(cfg.x_path).sort_values(["anonymized_id", "time_in_hour"])
    Y_raw = pd.read_parquet(cfg.y_path).sort_values(["anonymized_id", "time_in_hour"])

    # 2. Split
    x_tr_pd, x_va_pd, y_tr_pd, y_va_pd = chrono_split(X_raw, Y_raw, val_ratio=cfg.val_ratio)

    # Insert before section 3 in train_val
    raw_mid_price_tr = (y_tr_pd["ask_price_1"] + y_tr_pd["bid_price_1"]) / 2.0
    raw_mid_price_va = (y_va_pd["ask_price_1"] + y_va_pd["bid_price_1"]) / 2.0

    y_tr_pd = y_tr_pd.assign(mid_price=raw_mid_price_tr)
    y_va_pd = y_va_pd.assign(mid_price=raw_mid_price_va)

    # --- REPLACING SECTION 3: Preprocess with LOBProcessor ---
    processor = LOBProcessor(cfg, device=cfg.device)
    
    # 1. Process Train (calculates means/stds)
    train_out = processor.process(pl.from_pandas(x_tr_pd), pl.from_pandas(y_tr_pd))
    X_tr_tens, Y_tr_tens = train_out["X"], train_out["y"]
    
    # 2. Process Val (reuses train_out["means"] and train_out["stds"])
    val_out = processor.process(pl.from_pandas(x_va_pd), pl.from_pandas(y_va_pd))
    X_va_tens, Y_va_tens, y_va_map = val_out["X"], val_out["y"], val_out["y_id_map"]

    # 3. Enforce FEATURE_COLS order and extract scalers
    feat_indices = [processor.feature_map[col] for col in cfg.feature_cols]
    
    X_tr_tens = X_tr_tens[:, :, feat_indices]
    X_va_tens = X_va_tens[:, :, feat_indices]
    tr_means  = train_out["means"][:, :, feat_indices]
    tr_stds   = train_out["stds"][:, :, feat_indices]
    
    # Define indices for SuperModel and Target extraction
    new_f_map = {col: i for i, col in enumerate(cfg.feature_cols)}
    a_idx, b_idx = new_f_map["ask_price_1"], new_f_map["bid_price_1"]
    # -------------------------------------------------------

    # Extract normalized mid_price for training target
    mid_col_idx = new_f_map["mid_price"]
    Y_tr_tens = Y_tr_tens[:, :, feat_indices]
    Y_va_tens = Y_va_tens[:, :, feat_indices]

    mid_tr = Y_tr_tens[:, :, mid_col_idx]  # [60, num_ids]
    mid_va = Y_va_tens[:, :, mid_col_idx]  # [60, num_ids]

    # mid_tr has shape [60, Num_IDs] -> .T makes [Num_IDs, 60]
    train_ds = TensorTimeDataset(X_tr_tens, Y_tr_tens, mid_tr.T, input_window=cfg.input_window)
    val_ds   = TensorTimeDataset(X_va_tens, Y_va_tens, mid_va.T, input_window=cfg.input_window)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 6. Model & Training
    model = SuperModel(
        horizon=cfg.target_window, 
        ask_bid_idx=(a_idx, b_idx) # Use the indices relative to FEATURE_COLS
    ).to(cfg.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        tr_l = train_epoch(model, train_loader, optimizer, cfg)
        va_l = eval_epoch(model, val_loader, cfg)
        print(f"Epoch {epoch+1:02d} | Train: {tr_l:.6f} | Val: {va_l:.6f}")

    scalers = {
        "feat_means": tr_means,
        "feat_stds": tr_stds,
        "mid_mean": train_out["mid_mean"],       # per-instrument [1, num_train_ids], raw space
        "mid_stds": train_out["mid_stds"],       # per-instrument [1, num_train_ids], raw space
        "val_mid_mean": val_out["mid_mean"],     # per-instrument [1, num_val_ids], raw space
        "val_mid_stds": val_out["mid_stds"],     # per-instrument [1, num_val_ids], raw space
    }
    return model, scalers, val_loader, y_va_map, processor

def forecast_loss(pred, target, smooth_lambda=0.02):
    mse = (pred - target) ** 2
    loss = mse.mean()
    if smooth_lambda > 0:
        smooth = (pred[:, 1:] - pred[:, :-1]) ** 2
        loss = loss + smooth_lambda * smooth.mean()
    return loss
