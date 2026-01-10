import torch
import polars as pl

PRICE_COLS = [
    'ask_price_1','bid_price_1',
    'ask_price_2','bid_price_2',
    'ask_price_3','bid_price_3',
    'ask_price_4','bid_price_4',
    'ask_price_5','bid_price_5',
    'open','high','low','close',
]

VOL_COLS = [
    'ask_vol_1','bid_vol_1',
    'ask_vol_2','bid_vol_2',
    'ask_vol_3','bid_vol_3',
    'ask_vol_4','bid_vol_4',
    'ask_vol_5','bid_vol_5',
    'volume',
]

def backfill_first_nans(df: pl.DataFrame, cols: list[str]):
    for col in cols:
        first_valid_idx = (
            df
            .select((pl.col(col).is_not_null()).arg_max())
            .item()
        )

        if first_valid_idx == 0:
            continue  # no leading NaNs

        # Backfill only the leading NaNs
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) < first_valid_idx)
              .then(pl.lit(df[col][first_valid_idx]))
              .otherwise(pl.col(col))
              .alias(col)
        )
    return df

def df_to_tensor(df: pl.DataFrame, seq_len: int = 60, id_col: str = "anonymized_id", time_col: str = "time_in_hour") -> torch.Tensor:
    """
    Convert a Polars DataFrame into a tensor of shape (seq_len, num_ids, n_features).
    
    Args:
        df: Polars DataFrame containing ID, time, and feature columns.
        seq_len: Number of timesteps per sequence.
        id_col: Column name for the unique ID.
        time_col: Column name for the time dimension.
        
    Returns:
        X: Tensor of shape (seq_len, num_ids, n_features)
    """
    
    # Convert time column to int32
    df = df.with_columns([
        pl.col(time_col).cast(pl.Int32).alias(f"{time_col}_int")
    ])
    
    # Define feature columns
    time_int_col = f"{time_col}_int"
    feature_cols = [col for col in df.columns if col not in [id_col, time_col, time_int_col]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors
    features = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32).to(device)
    ids = torch.tensor(df[id_col].to_numpy(), dtype=torch.uint64).to(device)
    times = torch.tensor(df[time_int_col].to_numpy()).to(device)
    
    # Unique IDs
    unique_ids = torch.unique(ids)
    num_ids = len(unique_ids)
    n_features = features.shape[1]
    
    # Initialize tensor
    X = torch.zeros(seq_len, num_ids, n_features, dtype=torch.float32).to(device)

    id_map = list(enumerate(unique_ids))
    
    for i, uid in id_map:
        mask = ids == uid
        uid_features = features[mask]
        
        # Sort by time_in_hour if needed
        sorted_idx = torch.argsort(times[mask])
        uid_features = uid_features[sorted_idx]
        
        # Assign to tensor
        X[:, i, :] = uid_features  # assumes exactly seq_len timesteps per ID
    
    # Convert id_map to tensor for return (extract scalar values from tensors)
    id_map_data = [(i, uid.item()) for i, uid in id_map]
    id_map_tensor = torch.tensor(id_map_data, dtype=torch.uint64).to(device)
    return X, id_map_tensor

def preprocess(X, y, device) -> tuple:
    """
    Clean, validate, normalize, and tensorize feature and target datasets.

    This function applies a deterministic preprocessing pipeline to input
    feature (`X`) and target (`y`) Polars DataFrames. The pipeline enforces
    strict temporal consistency per instrument, handles missing values using
    market-aware rules, removes invalid instruments, converts data into
    fixed-length PyTorch tensors, and applies z-score normalization.

    Processing steps:
    1. Sort data by instrument ID and time.
    2. Backfill only *leading* NaNs in price columns per instrument.
    3. Forward-fill remaining price NaNs.
    4. Replace missing volume values with zeros.
    5. Remove instruments with:
       - duplicate (ID, time) pairs
       - missing timesteps (incomplete sequences)
    6. Convert DataFrames into tensors of shape:
       - X: (input_seq_len, num_ids, num_features)
       - y: (target_seq_len, num_ids, num_features)
    7. Compute per-ID, per-feature mean and standard deviation over time.
    8. Apply z-score normalization using statistics from X.

    Parameters
    ----------
    X : pl.DataFrame
        Input feature DataFrame containing price, volume, ID, and time columns.
    y : pl.DataFrame
        Target DataFrame with the same structure and alignment as X.

    Returns
    -------
    X : torch.Tensor
        Normalized input tensor of shape (input_seq_len, num_ids, num_features).
    y : torch.Tensor
        Normalized target tensor of shape (target_seq_len, num_ids, num_features).
    means : torch.Tensor
        Mean values used for normalization, shape (1, num_ids, num_features).
    stds : torch.Tensor
        Standard deviation values used for normalization,
        shape (1, num_ids, num_features).

    Notes
    -----
    - Normalization statistics are computed from X and reused for y.
    - All tensors are placed on the active PyTorch device (CPU or CUDA).
    - The function assumes exactly one observation per timestep per ID.
    """

    # 0. Sort
    X = X.sort(["anonymized_id", "time_in_hour"])
    y = y.sort(["anonymized_id", "time_in_hour"])

    # 1. Backfill *only* leading NaNs (prices only)
    X = (
        X
        .group_by("anonymized_id")
        .map_groups(lambda df: backfill_first_nans(df, PRICE_COLS))
    )
    y = (
        y
        .group_by("anonymized_id")
        .map_groups(lambda df: backfill_first_nans(df, PRICE_COLS))
    )

    # 2. Forward-fill the rest of price fields
    X = (
        X
        .group_by("anonymized_id")
        .map_groups(lambda df: df.with_columns(
            pl.col(PRICE_COLS).fill_null(strategy="forward")
        ))
    )   
    y = (
        y
        .group_by("anonymized_id")
        .map_groups(lambda df: df.with_columns(
            pl.col(PRICE_COLS).fill_null(strategy="forward")
        ))
    )

    # 3. Volume columns → 0
    X = X.with_columns(
        pl.col(VOL_COLS).fill_null(0)
    )
    y = y.with_columns(
        pl.col(VOL_COLS).fill_null(0)
    )   

    # drop time_in_hour within anonymized_id
    # Count rows per ID and time_in_hour (duplicates)
    dup_ids = (
        X
        .group_by(["anonymized_id", "time_in_hour"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)       # only keep duplicates
        .select("anonymized_id")
        .unique()
    )

    # Drop IDs with missing timesteps
    seq_len = 3600-60  # your expected sequence length

    # Count timesteps per ID
    valid_ids = (
        X
        .group_by("anonymized_id")
        .agg(pl.count("time_in_hour").alias("n_timesteps"))
        .filter(pl.col("n_timesteps") == seq_len)
        .select("anonymized_id")
    )

    # Remove duplicate ids and keep only IDs with full sequences -> 674 ids should be left
    X = X.filter(~pl.col("anonymized_id").is_in(dup_ids["anonymized_id"])).filter(pl.col("anonymized_id").is_in(valid_ids["anonymized_id"]))
    y = y.filter(~pl.col("anonymized_id").is_in(dup_ids["anonymized_id"])).filter(pl.col("anonymized_id").is_in(valid_ids["anonymized_id"]))


    y, y_id_map = df_to_tensor(y, seq_len=60, id_col="anonymized_id", time_col="time_in_hour")
    X, X_id_map = df_to_tensor(X, seq_len=3600-60, id_col="anonymized_id", time_col="time_in_hour")

    X, y, X_id_map, y_id_map = X.to(device), y.to(device), X_id_map.to(device), y_id_map.to(device)

    # compute z-score stats per ID + feature (across the sequence dimension)
    means = X.mean(dim=0, keepdim=True)    
    stds  = X.std(dim=0, keepdim=True)     

    # z-score normalize
    X = (X - means) / stds
    y = (y - means) / stds

    return X, y, means, stds, X_id_map, y_id_map
