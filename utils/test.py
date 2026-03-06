import torch
import pandas as pd
import polars as pl
from utils.datastuff import InferenceTensorDataset, LOBProcessor
from torch.utils.data import DataLoader

def generate_test_loader(cfg, processor):
    # 1. Load Data
    X_test_raw = pd.read_parquet(cfg.x_test_path).sort_values(["anonymized_id", "time_in_hour"])

    # Reuse the fitted processor (with training means/stds)
    test_out = processor.process(pl.from_pandas(X_test_raw), y_df=None)
    X_te_tens = test_out["X"]

    # 3. Enforce FEATURE_COLS order and extract scalers
    feat_indices = [processor.feature_map[col] for col in cfg.feature_cols]
    
    X_te_tens = X_te_tens[:, :, feat_indices]
    te_means  = test_out["means"][:, :, feat_indices]
    te_stds   = test_out["stds"][:, :, feat_indices]
    test_map = test_out["X_id_map"]
    
    test_ds = InferenceTensorDataset(X_te_tens, input_window=cfg.input_window)

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    scalers = {
        "feat_means": te_means, 
        "feat_stds": te_stds, 
    }
    return test_loader, scalers, test_map, processor

def generate_test_predictions(model, cfg, processor, num_ids=None):
    """
    Args:
        model: Trained model
        processor: The SAME LOBProcessor instance used for training (crucial for normalization!)
        cfg: Configuration object
        num_ids: (Optional) Int. If set, only predicts for this many anonymized_ids.
    """
    model.eval()
    
    # Use the processor passed in (with fitted means/stds from training)
    # Do NOT create a new one: processor = LOBProcessor(cfg)

    # 1. Load Test Data (Lazy Scan is faster for sampling)
    print("Loading test data...")
    x_test_lazy = pl.scan_parquet(cfg.x_path) 
    
    if num_ids is not None:
        print(f"Sampling first {num_ids} IDs for inference...")
        # Get the unique IDs and take the first N
        unique_ids = x_test_lazy.select("anonymized_id").unique().head(num_ids).collect()
        # Filter the lazy frame to only include these IDs
        x_test_pl = x_test_lazy.filter(pl.col("anonymized_id").is_in(unique_ids["anonymized_id"])).collect()
    else:
        x_test_pl = x_test_lazy.collect()
    
    # 2. Process with existing LOBProcessor
    print("Preprocessing test data...")
    test_out = processor.process(x_test_pl, y_df=None)
    X_test_tens = test_out["X"]
    test_id_map = test_out["X_id_map"] 
    
    # 3. Enforce Feature Order
    feat_indices = [processor.feature_map[col] for col in cfg.feature_cols]
    X_test_tens = X_test_tens[:, :, feat_indices]
    
    # 4. Inference
    print(f"Running inference on {X_test_tens.size(1)} instruments...")
    with torch.no_grad():
        # [3540, Num_IDs, Feat] -> [Num_IDs, 3540, Feat]
        model_input = X_test_tens.permute(1, 0, 2).to(cfg.device)
        preds = model(model_input) 
    
    preds_np = preds.cpu().numpy()
    return preds_np, test_id_map