import pandas as pd
from utils.datastuff import LOBProcessor

def generate_test_predictions(model, cfg):
    model.eval()

    processor = LOBProcessor(cfg)
    
    # 1. Load Test Data
    print("Loading test data...")
    x_test_pd = pd.read_parquet(cfg.x_path) # or wherever your test path is
    
    # 2. Process with existing LOBProcessor
    # Note: We pass None for y_df since test data doesn't have labels
    print("Preprocessing test data...")
    test_out = processor.process(pl.from_pandas(x_test_pd), y_df=None)
    X_test_tens = test_out["X"]
    test_id_map = test_out["X_id_map"] # Needed to link predictions back to IDs
    
    # 3. Enforce Feature Order
    # Use the same indices used during training
    feat_indices = [processor.feature_map[col] for col in cfg.feature_cols]
    X_test_tens = X_test_tens[:, :, feat_indices]
    
    # 4. Inference
    print(f"Running inference on {X_test_tens.size(1)} instruments...")
    with torch.no_grad():
        # Input shape: [3540, Num_IDs, 20] 
        # Most models expect [Batch/IDs, Seq, Feat], so we transpose:
        # permute(1, 0, 2) -> [Num_IDs, 3540, 20]
        model_input = X_test_tens.permute(1, 0, 2).to(cfg.device)
        
        # Generate predictions [Num_IDs, 60]
        preds = model(model_input) 
    
    # 5. Bring back to CPU/Numpy for submission
    preds_np = preds.cpu().numpy()
    
    return preds_np, test_id_map