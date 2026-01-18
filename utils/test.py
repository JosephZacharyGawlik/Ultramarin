import torch
import pandas as pd
import polars as pl
from utils.datastuff import LOBProcessor

def generate_test_predictions(model, cfg, num_ids=None):
    """
    Args:
        model: Trained model
        processor: The SAME LOBProcessor instance used for training (crucial for scaling!)
        cfg: Configuration object
        num_ids: (Optional) Int. If set, only predicts for this many anonymized_ids.
    """
    model.eval()

    processor = LOBProcessor(cfg)

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