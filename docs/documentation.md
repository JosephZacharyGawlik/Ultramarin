# Execution Strategy Evolution

## Table of Contents
1. [Objective & Metric](#objective--metric)
2. [Baselines](#baselines)
3. [Ridge Regression](#ridge-regression)
4. [TCN (Neural)](#tcn-neural)
5. [Current Best Results](#current-best-results)
6. [Lessons & Next Steps](#lessons--next-steps)

---

## Objective & Metric
- **Goal:** Fill a fixed volume in the last minute (59:00–59:59) of each hour while minimizing implementation error.
- **Implementation error:**  
  `|avg_price – close_price| / close_price × fill_penalty × 10,000 (bps)`  
  where `fill_penalty = min(100, volume_to_fill / total_volume_executed)`.
- **Execution simulation:** `simulate_walk_the_book` consumes quoted depth per second and returns executed volume and VWAP.

---

## Baselines
### TWAP / Uniform
- Spread volume evenly across all 60 seconds (full-hour TWAP) or only the last *K* seconds (uniform-in-window).
- Used as a dumb benchmark to beat.

### Fixed End-of-Hour (Last K Seconds)
- Trade only in the final *K* seconds (e.g., 14–20), equal size per second.
- Motivation: prices near the close tend to hug the official close price, reducing the metric.

### Liquidity-Driven (“Smart”)
- Trade only in the last *K* seconds (default 14).
- Sequential, causal: at each second, scale the base fill by current top-of-book liquidity vs the previous second (capped multipliers), no look-ahead, no post-hoc rescaling.
- Best heuristic baseline so far (~1.32 bps on BTCUSDT validation).

---

## Ridge Regression
**Purpose:** Cheap, interpretable baseline to see how far hour-level summaries can forecast the last minute.

**Features (hour-level aggregates over 00:00–58:59):**
- Summaries of price and book shape (e.g., where mid and spread started vs ended, average/tail spreads and depth, overall imbalance/range).
- Summaries of activity (e.g., total/typical volume).
- Same style of stats over short trailing windows (last 60s/5m/15m) to capture recent regimes.
*Example:* “Average spread and depth over the hour” plus “spread/depth/imbalance over the last minute” instead of raw per-second data.

**Targets (last minute summaries, one per hour):**
- `target_close`, `target_mean_spread`, `target_mean_bid_vol`, `target_mean_ask_vol`, `target_mean_top_depth`.

**Training & Scheduling:**
- Chronological split (~80/20). Features standardized; multi-target ridge (L2, bias unregularized).
- Predictions steer a simple scheduler: choose how early to enter the last minute and how aggressively to size fills, based on predicted close/average spread/liquidity. Generate a 60-second position vector from those heuristics and score via `compute_implementation_error`.

**Results (BTCUSDT val):**
- Close MAE ≈ 631; spread MAE ≈ 2.84; level-1 vol MAE ≈ 0.11–0.18.
- Implementation error: ~1.43 bps vs uniform 1.72, end-of-hour 1.37.

**How to explain it (and why it’s limited):**
- It collapses 59 minutes of second-level data into a handful of hour-level stats, then tries to guess how the last minute will “look on average” (close, average spread/liquidity).
- It’s intentionally crude—a sanity baseline to benchmark against—not a production scheduler. Because it throws away per-second structure, it can’t tell you which exact seconds to trade and struggles on the small-scale liquidity signals that drive execution quality.

---

## TCN (Neural)
**Purpose:** Learn per-second structure leading into the execution minute to time fills better.

**Inputs (per-second, last 5 minutes before 59:00):**
- Midprice, spread, spread%, level-1..5 volumes, total depth, imbalance, volume, 1s mid returns, 5s rolling means (volume/spread/imbalance).
- Features normalized (mean/std); NaNs filled.

**Target (final config):**
- Per-second midprice for 59:00–59:59 (60×1), z-scored on the train split. We dropped spread/liquidity targets after they proved noisy and hurt scheduling.

**Model & Training:**
- TCN with dilated residual Conv1d blocks; output reshaped to 60×1.
- Loss: weighted MSE (midprice only) + smoothing penalty on successive timesteps (default smooth_lambda=0.02) to stabilize the predicted path.

**Scheduler (final config):**
1. Score seconds by proximity of predicted mid to predicted close (last predicted mid); cap scores at 3× mean to avoid over-concentration.
2. Convert scores to a model schedule (weights × volume_to_fill).
3. Build the smart liquidity schedule (last 14s, scaled by observed liquidity).
4. Blend schedules: 20% model + 80% liquidity baseline; rescale to exactly volume_to_fill per hour. Fall back to uniform if needed.
5. Submit 60-second positions to `simulate_walk_the_book` and score.

**Results (BTCUSDT val):**
- Midprice forecasts: MAE ≈ 2,016; RMSE ≈ 3,181; R² ≈ 0.99.
- Implementation error: ~1.30 bps vs uniform 1.72, end-of-hour 1.37, liquidity baseline 1.32 (best current configuration).

**Notes:**
- Many NaN warnings from `simulate_walk_the_book` come from missing quote levels in `y_train`; those levels are skipped and do not abort the run.

---

## Current Best Results
- **Liquidity baseline:** ~1.32 bps (heuristic best).
- **TCN midprice-only + blend:** ~1.30 bps (best overall on validation subset).
- **End-of-hour (fixed window):** ~1.37 bps.
- **Uniform/TWAP:** ~1.72 bps.
- **Ridge-guided:** ~1.43 bps.

---

## Lessons & Next Steps
- Midprice forecasting is strong; liquidity/spread forecasts remain noisy and can hurt scheduling.
- Blending model timing with a liquidity-aware baseline gives robustness and improves over pure heuristics.
- Further gains would likely need either:
  - A coarse liquidity head/classifier (good/normal/bad) instead of raw volume regression, or
  - A more sophisticated allocator (e.g., dynamic programming over predicted prices with hard caps) while preserving causality.
- If expanding models, keep targets normalized, add per-channel heads if reintroducing liquidity, and enforce smoothness/score caps to avoid erratic allocations.
