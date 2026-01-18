import torch
import numpy as np
import pandas as pd
import polars as pl
from models.dataclasses import *
from models.Seq2SeqAttention import *
from utils.utils import preprocess
from torch.utils.data import DataLoader

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
    for xb, _, target in loader:
        xb, target = xb.to(cfg.device), target.to(cfg.device)
        opt.zero_grad()
        
        # SuperModel handles teacher forcing internally if y_teacher is provided
        pred = model(xb, y_teacher=target)
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
    raw_mid_price = (y_tr_pd["ask_price_1"] + y_tr_pd["bid_price_1"]) / 2.0
    actual_mid_mean = raw_mid_price.mean()
    actual_mid_std = raw_mid_price.std()

    # 3. Preprocess (to Polars -> to Tensors)
    X_tr_tens, Y_tr_tens, tr_means, tr_stds, _, _, f_map = preprocess(
        pl.from_pandas(x_tr_pd), pl.from_pandas(y_tr_pd), cfg.device
    )
    X_va_tens, Y_va_tens, _, _, _, y_va_map, _ = preprocess(
        pl.from_pandas(x_va_pd), pl.from_pandas(y_va_pd), cfg.device
    )

    # --- THE FIX: ENFORCE FEATURE_COLS ORDER ---
    # This ensures index 0-19 match the [P, V, P, V] pattern DeepLOB expects
    feat_indices = [f_map[col] for col in cfg.feature_cols]
    
    X_tr_tens = X_tr_tens[:, :, feat_indices]
    X_va_tens = X_va_tens[:, :, feat_indices]
    
    # Also re-order the means/stds so they match the new tensor indices
    tr_means = tr_means[:, :, feat_indices]
    tr_stds  = tr_stds[:, :, feat_indices]
    # -------------------------------------------
    
    # 4. Target Mid-Price extraction
    # Note: Because we re-ordered above, ask_price_1 is now index 0, bid_price_1 is index 2
    # But using the re-mapped indices via f_map logic is still safer:
    new_f_map = {col: i for i, col in enumerate(cfg.feature_cols)}
    a_idx, b_idx = new_f_map["ask_price_1"], new_f_map["bid_price_1"]
    
    mid_tr = (Y_tr_tens[:, :, f_map["ask_price_1"]] + Y_tr_tens[:, :, f_map["bid_price_1"]]) / 2.0
    mid_va = (Y_va_tens[:, :, f_map["ask_price_1"]] + Y_va_tens[:, :, f_map["bid_price_1"]]) / 2.0

    mid_mean = float(mid_tr.mean())
    mid_std  = float(mid_tr.std()) if mid_tr.std() != 0 else 1e-6

    # 5. Datasets (passing cfg.input_window)
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
        "mid_mean": actual_mid_mean, # Use raw dollar values
        "mid_std": actual_mid_std     # Use raw dollar values
    }
    return model, scalers, val_loader, y_va_map

def forecast_loss(pred, target, smooth_lambda=0.02):
    mse = (pred - target) ** 2
    loss = mse.mean()
    if smooth_lambda > 0:
        smooth = (pred[:, 1:] - pred[:, :-1]) ** 2
        loss = loss + smooth_lambda * smooth.mean()
    return loss
