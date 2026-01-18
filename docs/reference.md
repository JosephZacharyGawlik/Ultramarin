# Quick Reference Guide

## Strategy Performance

| Strategy | Error (bps) | Status |
|----------|-------------|--------|
| Uniform | 1.5121 | Baseline |
| End-of-hour (14s) | 1.2186 | ✅ Good |
| Smart end-of-hour | 1.0848-1.1691 | ✅ Best |

## Key Functions

### Evaluation
```python
from utils import compute_implementation_error

error = compute_implementation_error(
    y_submission=submission_df,
    y_train=y_train_df,
    volume_to_fill=4.0
)
# Returns: error in basis points (bps)
```

### Create Submission Template
```python
from utils import create_submission_template

template = create_submission_template(y_train)
# Returns: DataFrame with all (anonymized_id, time_in_hour) combinations
```

### Baseline Strategies
```python
from utils import baseline_uniform, baseline_end_of_hour, baseline_smart_end_of_hour

# Uniform distribution
submission1 = baseline_uniform(y_train, volume_to_fill=4.0)

# End-of-hour (last 14 seconds)
submission2 = baseline_end_of_hour(y_train, volume_to_fill=4.0, seconds_from_end=14)

# Smart end-of-hour (liquidity-aware)
submission3 = baseline_smart_end_of_hour(y_train, volume_to_fill=4.0)
```

## Submission Format

```python
submission = pl.DataFrame({
    'anonymized_id': [id1, id1, ..., id2, id2, ...],  # 60 per ID
    'time_in_hour': [0s, 1s, ..., 59s, 0s, 1s, ...],  # Duration format
    'position': [0.067, 0.067, ..., 0.067, ...]  # Sum = volume_to_fill per ID
})
```

## Key Insights

1. **Metric**: `|avg_price - close_price| / close_price` (minimize)
2. **Bonus**: `(avg_price - close_price) / close_price` (negative for buying = good!)
3. **Timing**: End-of-hour works (prices converge to close_price)
4. **No data leakage**: Make decisions sequentially, use only past data
5. **Full fill**: Must fill exactly `volume_to_fill` per hour

## Common Mistakes to Avoid

❌ Using future information for normalization  
❌ Optimizing execution cost instead of matching close_price  
❌ Trading too early (prices drift from close_price)  
❌ Using close_price in strategy (not available at execution time!)

## Data Structure

- **X_train**: First 59 minutes (00:00-58:59)
- **y_train**: Last 1 minute (59:00-59:59) + close prices
- **X_test**: Test data (y_test is hidden!)
- **705 hours** per cryptocurrency pair
- **60 seconds** per minute

## Order Book Levels

- Level 1: Best bid/ask (closest to mid price)
- Level 2-5: Deeper levels
- `ask_price_1` = best ask price (lowest)
- `bid_price_1` = best bid price (highest)
- `ask_vol_1` = volume at best ask
- `bid_vol_1` = volume at best bid

## Execution Flow

1. Submit positions for 60 seconds (59:00-59:59)
2. `simulate_walk_the_book()` executes against order book
3. Unfilled positions carry forward to next second
4. Calculate VWAP: `total_value / total_volume`
5. Compare to close_price: `|VWAP - close_price| / close_price`

