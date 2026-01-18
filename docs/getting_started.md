# Getting Started Guide: The Execution Edge Challenge

## Overview

This is an algorithmic trading execution optimization challenge. Your goal is to minimize **implementation error** (execution cost) when trading cryptocurrency by optimizing when and how much to trade within each hour.

## Quick Start

1. **Open the getting_started.ipynb notebook** - This contains step-by-step code to explore the data and set up baselines.

2. **Understand the data structure**:
   - `X_train.parquet`: Order book data with ask/bid prices and volumes at 5 levels
   - `y_train.parquet`: Target data with anonymized_id, time_in_hour, and close prices
   - `vol_to_fill.txt`: Target volume to execute (e.g., 4.00 for BTCUSDT)

3. **Set up evaluation**: Use the `compute_implementation_error()` function from `utils.py` to evaluate your submissions.

4. **Start with baselines**: 
   - Uniform distribution: Spread volume evenly across all 60 seconds
   - End-of-hour: Fill only in the last N seconds (often performs better!)

## Key Concepts

### Submission Format
Your submission must be a DataFrame with exactly 3 columns:
- `anonymized_id`: Unique identifier for each trading hour
- `time_in_hour`: Time within the hour (duration format)
- `position`: Trade size (positive = buy, negative = sell)

**Important**: The sum of positions for each anonymized_id must equal `volume_to_fill`.

### Scoring Metric
You minimize:
```
|average_price - close_price| / close_price × min(100, volume_to_fill / total_volume_executed)
```

Where:
- `close_price`: Last traded price at end of hour
- `average_price`: Volume-weighted average execution price from walking the order book
- Penalty: Multiplicative penalty if you don't fill the full volume

### Execution Simulation
The `simulate_walk_the_book()` function simulates how your orders execute:
- Positive positions = buy orders (consume ask side)
- Negative positions = sell orders (consume bid side)
- Unfilled positions carry forward to the next second
- Walks through multiple price levels until filled or liquidity exhausted

## Step-by-Step Approach

### Phase 1: Data Exploration
1. Load and explore the data structure
2. Understand the order book features (5 levels of ask/bid prices and volumes)
3. Analyze spread patterns and liquidity throughout the hour
4. Check the relationship between order book depth and execution cost

### Phase 2: Baseline Models
1. **Uniform baseline**: Distribute volume evenly (score ~1.512 bps)
2. **End-of-hour baseline**: Fill only near the end (score ~1.218 bps)
3. **Spread benchmark**: ~1.206 bps (try to beat this!)

### Phase 3: Feature Engineering
Key features to explore:
- Spread percentage
- Mid price
- Best bid/ask liquidity
- Depth imbalance (ask_vol vs bid_vol)
- Total depth across 5 levels
- Price levels relative to mid

### Phase 4: Strategy Development
Ideas to try:
1. **Timing optimization**: Find optimal time windows for execution
2. **Liquidity-aware**: Trade when liquidity is high
3. **Spread-aware**: Avoid trading when spread is wide
4. **Price prediction**: Predict close price and optimize accordingly
5. **Machine learning**: Train models to predict optimal positions

## Files Overview

- `getting_started.ipynb`: Step-by-step tutorial notebook
- `utils.py`: Utility functions for data loading, evaluation, and baselines
- `simulate_walk_the_book.py`: Order book execution simulation (already provided)
- `example.ipynb`: Original example notebook from the challenge

## Tips

1. **Start small**: Begin with BTCUSDT only, then expand to other pairs
2. **Validate locally**: Use `compute_implementation_error()` to test before submitting
3. **Watch the fill penalty**: Make sure you're filling the full volume!
4. **Consider both sides**: Try both buy (positive) and sell (negative) strategies
5. **Cross-validation**: Split your data properly to avoid overfitting

## Next Steps

1. Run through `getting_started.ipynb` to understand the data
2. Evaluate the baseline models to establish benchmarks
3. Explore feature engineering opportunities
4. Develop and test your strategy
5. Optimize and iterate!

Good luck! 🚀

