import torch
import pandas as pd
import polars as pl
from utils.datastuff import InferenceTensorDataset, LOBProcessor
from torch.utils.data import DataLoader

def generate_test_loader(cfg, processor):
    # 1. Load Data
    X_test_raw = pd.read_parquet(cfg.x_test_path).sort_values(["anonymized_id", "time_in_hour"])

    # Add mid_price so test data has the same features as training data
    X_test_raw = X_test_raw.assign(mid_price=(X_test_raw["ask_price_1"] + X_test_raw["bid_price_1"]) / 2.0)

    # Reuse the fitted processor (with training means/stds)
    test_out = processor.process(pl.from_pandas(X_test_raw), y_df=None)
    X_test_tensor = test_out["X"]

    # Enforce FEATURE_COLS order and extract scalers
    feature_indices = [processor.feature_map[col] for col in cfg.feature_cols]

    X_test_tensor = X_test_tensor[:, :, feature_indices]
    test_means    = test_out["means"][:, :, feature_indices]
    test_stds     = test_out["stds"][:, :, feature_indices]
    test_id_map   = test_out["X_id_map"]

    test_dataset = InferenceTensorDataset(X_test_tensor, input_window=cfg.input_window)
    test_loader  = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    scalers = {
        "feat_means": test_means,
        "feat_stds": test_stds,
    }
    return test_loader, scalers, test_id_map, processor

def generate_test_predictions(model, cfg, processor, num_ids=None):
    """
    Args:
        model: Trained model
        processor: The SAME LOBProcessor instance used for training (crucial for normalization!)
        cfg: Configuration object
        num_ids: (Optional) Int. If set, only predicts for this many anonymized_ids.
    """
    model.eval()

    # 1. Load Test Data (Lazy Scan is faster for sampling)
    print("Loading test data...")
    x_test_lazy = pl.scan_parquet(cfg.x_path)

    if num_ids is not None:
        print(f"Sampling first {num_ids} IDs for inference...")
        unique_ids = x_test_lazy.select("anonymized_id").unique().head(num_ids).collect()
        x_test_polars = x_test_lazy.filter(pl.col("anonymized_id").is_in(unique_ids["anonymized_id"])).collect()
    else:
        x_test_polars = x_test_lazy.collect()

    # Add mid_price so test data has the same features as training data
    x_test_polars = x_test_polars.with_columns(
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid_price")
    )

    # 2. Process with existing LOBProcessor
    print("Preprocessing test data...")
    test_out = processor.process(x_test_polars, y_df=None)
    X_test_tensor = test_out["X"]
    test_id_map = test_out["X_id_map"]

    # 3. Enforce Feature Order
    feature_indices = [processor.feature_map[col] for col in cfg.feature_cols]
    X_test_tensor = X_test_tensor[:, :, feature_indices]

    # 4. Inference
    print(f"Running inference on {X_test_tensor.size(1)} instruments...")
    with torch.no_grad():
        # [3540, Num_IDs, Feat] -> [Num_IDs, 3540, Feat]
        model_input = X_test_tensor.permute(1, 0, 2).to(cfg.device)
        predictions = model(model_input)

    predictions_np = predictions.cpu().numpy()
    return predictions_np, test_id_map
