"""
Utility functions for The Execution Edge challenge.
"""
import polars as pl
import numpy as np
import re
from pathlib import Path
from typing import Tuple
from simulate_walk_the_book import simulate_walk_the_book


def load_data(data_dir: str) -> Tuple[pl.DataFrame, pl.DataFrame, float]:
    """
    Load training data and volume to fill.
    
    Args:
        data_dir: Directory containing X_train.parquet, y_train.parquet, vol_to_fill.txt
    
    Returns:
        Tuple of (X_train, y_train, volume_to_fill)
    """
    X_train = pl.read_parquet(f"{data_dir}/X_train.parquet")
    y_train = pl.read_parquet(f"{data_dir}/y_train.parquet")
    
    # Load volume to fill
    with open(f"{data_dir}/vol_to_fill.txt", "r") as file:
        content = file.read().strip()
    match = re.search(r"The volume to fill is: ([\d.]+)", content)
    volume_to_fill = float(match.group(1)) if match else None
    
    return X_train, y_train, volume_to_fill


def create_submission_template(y_train: pl.DataFrame) -> pl.DataFrame:
    """Create a template with all anonymized_ids and time_in_hour combinations."""
    all_time_in_hour = y_train['time_in_hour'].unique().sort()
    unique_ids = y_train['anonymized_id'].unique().sort()
    
    templated_df = unique_ids.to_frame().join(
        all_time_in_hour.to_frame(),
        how="cross"
    )
    
    return templated_df


def compute_implementation_error(
    y_submission: pl.DataFrame,
    y_train: pl.DataFrame,
    volume_to_fill: float
) -> float:
    """
    Compute implementation error for a submission.
    
    Args:
        y_submission: DataFrame with columns [anonymized_id, time_in_hour, position]
                     Contains positions for the last 60 seconds of each hour
        y_train: DataFrame with order book data (ask/bid prices/volumes) and close prices
                 for the last minute of each hour. This has everything we need!
        volume_to_fill: Target volume to execute
    
    Returns:
        Mean implementation error in basis points
    
    Formula:
        error = |avg_price - close_price| / close_price × min(100, volume_to_fill / total_vol) × 10000
    """
    # Join submission with y_train to get order book data for each position
    # y_train contains: order book data (ask/bid prices/volumes) + close prices
    merged = (
        y_submission
        .join(y_train, on=['anonymized_id', 'time_in_hour'], how='left')
        .sort(['anonymized_id', 'time_in_hour'])
    )
    
    errors = []
    
    for anonymized_id in merged['anonymized_id'].unique():
        id_data = merged.filter(pl.col('anonymized_id') == anonymized_id)
        
        # Get close price from y_train - the LAST close price in the last minute
        # Formula from example.ipynb: y_last_min.select("close").drop_nulls().to_series().last()
        # This is the close price at the END of the hour (last second of last minute)
        close_prices = id_data['close'].drop_nulls()
        if len(close_prices) == 0:
            continue
        close_price = close_prices[-1]  # Get the LAST close price (end of hour)
        
        # Prepare order book arrays
        positions = id_data['position'].fill_null(0.0).to_numpy()
        
        # Extract ask/bid prices and volumes (5 levels)
        ask_prices = np.column_stack([
            id_data[f'ask_price_{i}'].fill_null(np.nan).to_numpy() 
            for i in range(1, 6)
        ])
        ask_vols = np.column_stack([
            id_data[f'ask_vol_{i}'].fill_null(np.nan).to_numpy() 
            for i in range(1, 6)
        ])
        bid_prices = np.column_stack([
            id_data[f'bid_price_{i}'].fill_null(np.nan).to_numpy() 
            for i in range(1, 6)
        ])
        bid_vols = np.column_stack([
            id_data[f'bid_vol_{i}'].fill_null(np.nan).to_numpy() 
            for i in range(1, 6)
        ])
        
        # Simulate walk the book
        total_vol, avg_price = simulate_walk_the_book(
            positions, ask_prices, ask_vols, bid_prices, bid_vols
        )
        
        if total_vol == 0 or np.isnan(avg_price) or np.isnan(close_price):
            continue
        
        # Compute implementation error
        relative_error = abs(avg_price - close_price) / close_price
        fill_penalty = min(100, volume_to_fill / total_vol) if total_vol > 0 else 100
        
        error_bps = relative_error * fill_penalty * 10000  # Convert to basis points
        errors.append(error_bps)
    
    return np.mean(errors) if errors else np.nan


def compute_features(X_data: pl.DataFrame) -> pl.DataFrame:
    """Compute additional features from order book data."""
    features = X_data.with_columns([
        # Spread features
        ((pl.col('ask_price_1') - pl.col('bid_price_1')) / 
         ((pl.col('ask_price_1') + pl.col('bid_price_1')) / 2)).alias('spread_pct'),
        
        # Mid price
        ((pl.col('ask_price_1') + pl.col('bid_price_1')) / 2).alias('mid_price'),
        
        # Total liquidity at best bid/ask
        (pl.col('ask_vol_1') + pl.col('bid_vol_1')).alias('best_liquidity'),
        
        # Depth imbalance
        ((pl.col('ask_vol_1') - pl.col('bid_vol_1')) / 
         (pl.col('ask_vol_1') + pl.col('bid_vol_1'))).alias('depth_imbalance'),
        
        # Total depth across 5 levels
        (pl.col('ask_vol_1') + pl.col('ask_vol_2') + pl.col('ask_vol_3') + 
         pl.col('ask_vol_4') + pl.col('ask_vol_5')).alias('total_ask_depth'),
        (pl.col('bid_vol_1') + pl.col('bid_vol_2') + pl.col('bid_vol_3') + 
         pl.col('bid_vol_4') + pl.col('bid_vol_5')).alias('total_bid_depth'),
        
        # Price levels relative to mid
        ((pl.col('ask_price_1') - (pl.col('ask_price_1') + pl.col('bid_price_1')) / 2) /
         ((pl.col('ask_price_1') + pl.col('bid_price_1')) / 2)).alias('ask_premium'),
        (((pl.col('ask_price_1') + pl.col('bid_price_1')) / 2) - pl.col('bid_price_1')) /
         ((pl.col('ask_price_1') + pl.col('bid_price_1')) / 2)).alias('bid_discount'),
    ])
    
    return features


def baseline_uniform(y_train: pl.DataFrame, volume_to_fill: float) -> pl.DataFrame:
    """Distribute volume uniformly across all seconds."""
    template = create_submission_template(y_train)
    
    submission = template.with_columns(
        position=pl.lit(volume_to_fill / 60)
    )
    
    return submission


def baseline_end_of_hour(y_train: pl.DataFrame, volume_to_fill: float, 
                        seconds_from_end: int = 14) -> pl.DataFrame:
    """Fill volume only in the last N seconds of the hour."""
    template = create_submission_template(y_train)
    
    # Get the last time_in_hour value
    last_time = y_train['time_in_hour'].max()
    
    # Create threshold (last_time - seconds_from_end seconds)
    threshold = last_time - pl.duration(seconds=seconds_from_end)
    
    submission = template.with_columns(
        position=pl.when(pl.col('time_in_hour') >= threshold)
        .then(pl.lit(volume_to_fill / seconds_from_end))
        .otherwise(pl.lit(0.0))
    )
    
    return submission

