# Execution Edge Challenge Writeup

## 1. Problem Statement

The goal of this challenge is to develop an optimal execution strategy for liquidating a fixed volume of an asset (e.g., BTCUSDT) within the **last 60 seconds** of a trading hour.

**Objective**: Minimize the **Implementation Shortfall** relative to the **Close Price** of that hour.
- **Benchmark**: The official Close Price (price at the exact end of the hour).
- **Execution Price**: The volume-weighted average price (VWAP) of our trades.
- **Metric**: Implementation Error in Basis Points (bps).
  $$ \text{Error (bps)} = \frac{|\text{Avg Price} - \text{Close Price}|}{\text{Close Price}} \times 10000 \times \text{Penalty} $$
- **Constraint**: The full volume must be executed within the 60-second window. Any unexecuted volume incurs a heavy penalty.

**Key Challenge**: We do not know the Close Price until the hour ends. We must trade *sequentially* using only past data (Limit Order Book state) to estimate where the price will settle and manage liquidity accordingly.

---

## 2. Baseline Models

We established several baseline strategies to benchmark our Deep Learning models against. These strategies use heuristic logic without complex predictive modeling.

### A. Uniform TWAP (Time-Weighted Average Price)
- **Logic**: Splits the total volume equally across all 60 seconds of the final minute.
- **Pros**: Simple, minimizes market impact per second.
- **Cons**: Ignores price trends and the fact that price volatility typically decreases/converges closer to the close.
- **Performance**: ~1.51 bps (Baseline).

### B. End-of-Hour Execution (14s Window)
- **Logic**: Waits until the last 14 seconds of the minute to start trading, then executes uniformly.
- **Hypothesis**: Prices closer to the end of the hour are statistically closer to the final Close Price.
- **Pros**: reduces tracking error by trading closer to the benchmark time.
- **Cons**: High concentration of volume in a short window raises liquidity costs (market impact).
- **Performance**: ~1.22 bps. (Significant improvement over Uniform).

### C. "Smart" End-of-Hour (Liquidity & Spread Aware)
- **Logic**: 
    1.  Trades only in the last **14 seconds**.
    2.  **Adaptive Sizing**: Allocates more volume to seconds with **higher liquidity** (Volume at Best Bid/Ask) relative to the recent past.
    3.  **Sequential Decision**: Does *not* peek into the future; decisions are made second-by-second based on the current LOB state.
- **Performance**: ~1.20 bps. (Slight improvement over standard End-of-Hour).

### D. Ridge Regression Baseline (`src/baselines.py`)
- **Model**: A Multi-Target Ridge Regression model.
- **Features**: Hand-engineered features from the first 59 minutes (e.g., average spread, order book imbalance, midprice returns, volatility).
- **Targets**: Predicts 5 key metrics for the final minute:
    1.  `target_close`
    2.  `target_mean_spread`
    3.  `target_mean_bid_vol`
    4.  `target_mean_ask_vol`
    5.  `target_mean_top_depth`
- **Strategy**: Uses the predicted `target_close` to adjust execution aggressiveness (Trade more if current price is "better" than predicted close).
- **Performance**: ~1.20 - 1.25 bps (Comparable to heuristics, proving that simple linear features struggle to beat the "wait until the end" heuristic).
