# Feature Notes For Joe

The goal here is just to make the feature side of the approach easier to follow:

- what the feature names mean
- what kinds of features we considered
- how we went from many candidate features to one deployed feature per adaptive pair
- how the bins were chosen

## 1. The Basic Idea

For each hour, we reduce the first 59 minutes to a small set of summary statistics.

Then, for adaptive pairs:

- choose one feature for that pair
- split that feature into a small number of bins
- map each bin to a predefined execution rule

So the model is not:

- "predict the close"

It is closer to:

- "use one summary of the first 59 minutes to decide which schedule this hour should get"

## 2. Naming Convention

Most feature names follow a simple pattern:

- `window` + `_` + `quantity` + `_` + `summary`

### Windows

- `last_...`
  - last visible snapshot at `58:59`
- `w60_...`
  - last visible `60` seconds
- `w300_...`
  - last visible `5` minutes
- `w900_...`
  - last visible `15` minutes
- `w1800_...`
  - last visible `30` minutes
- `w3540_...`
  - full visible `59` minutes

### Common summaries

- `mean`
  - average over the window
- `std`
  - standard deviation over the window
- `sum`
  - sum over the window
- `realized_vol`
  - realized volatility over the window
- `abs_return_sum`
  - sum of absolute returns over the window
- `row_count`
  - number of observed rows in the window
- `coverage`
  - fraction of the window that is observed
- `close_missing_rate`
  - fraction of timestamps where `close` is missing
- `volume_missing_rate`
  - fraction of timestamps where `volume` is missing

## 3. The Main Feature Families

We searched over single features from a few broad families.

### A. Spread features

These describe how tight or wide the market has been.

Examples:

- `last_rel_spread_1`
- `w60_rel_spread_mean`
- `w300_rel_spread_mean`
- `w900_rel_spread_std`

How to interpret them:

- low spread = tighter market
- high spread = wider market
- high spread volatility = market tightness is unstable

### B. Trade activity features

These describe how much trading has been happening.

Examples:

- `last_trade_volume`
- `w60_trade_volume_mean`
- `w300_trade_volume_sum`
- `w1800_trade_volume_mean`
- `w3540_trade_volume_mean`

How to interpret them:

- higher values mean more trade activity
- lower values mean quieter trading

### C. Visible depth / liquidity features

These describe the displayed size available in the book.

Examples:

- `last_ask_vol_1`
- `last_bid_vol_1`
- `w60_ask_vol_1_mean`
- `w300_top_bid_vol_5_mean`

Some of these also appear with `_to_fill`, meaning:

- divide visible size by `vol_to_fill`

That makes the feature relative to how much we actually need to execute.

### D. Imbalance features

These describe whether the visible book is more bid-heavy or ask-heavy.

Examples:

- `last_top_vol_imbalance_5`
- `w60_top_vol_imbalance_5_mean`
- `w300_top_vol_imbalance_5_mean`

### E. Return / volatility features

These describe how noisy or directional the recent price path has been.

Examples:

- `w300_abs_return_sum`
- `w900_realized_vol`
- `w3540_mid_return_mean`

How to interpret them:

- large absolute-return / volatility values mean the market has been moving around a lot

### F. Coverage / missingness features

These describe how complete or sparse the visible stream has been.

Examples:

- `w900_close_missing_rate`
- `w300_volume_missing_rate`
- `w900_coverage`
- `w900_row_count`

These ended up mattering especially for some of the noisier / thinner pairs.

## 4. Examples Of What The Selected Features Mean

Here are the final current adaptive features and the plain-English meaning of each.

### Ask side

| Pair | Feature | Plain-English Meaning |
| --- | --- | --- |
| `BTCUSDT` ask | `w300_rel_spread_mean` | average relative spread over the last 5 visible minutes; basically how tight the market has recently been |
| `DOGEUSDT` ask | `w900_close_missing_rate` | fraction of missing `close` values over the last 15 visible minutes; acts as a proxy for activity / coverage / sparsity |
| `ETHUSDT` ask | `w60_trade_volume_mean` | average trade volume over the last visible minute; how active trading has been very recently |
| `LTCUSDT` ask | `w300_abs_return_sum` | cumulative absolute return over the last 5 visible minutes; how jumpy/noisy price has been |
| `SOLUSDT` ask | `w900_rel_spread_std` | spread variability over the last 15 visible minutes; whether market tightness has been stable or unstable |

### Bid side

| Pair | Feature | Plain-English Meaning |
| --- | --- | --- |
| `BTCUSDT` bid | `w1800_trade_volume_mean` | average trade volume over the last 30 visible minutes; medium-term activity level |
| `DOGEUSDT` bid | `w300_close_missing_rate` | missing `close` rate over the last 5 visible minutes; short-term coverage / sparsity proxy |
| `ETHUSDT` bid | `w3540_trade_volume_mean` | average trade volume across the full visible 59 minutes; overall hourly activity level |

Pairs that currently use fixed rules instead of adaptive features:

- `ADAUSDT` ask: `fixed_k`
- `ADAUSDT` bid: `fixed_k`
- `LTCUSDT` bid: `fixed_k_alpha`
- `SOLUSDT` bid: `fixed_k`
- `XRPUSDT` ask: `fixed_k_alpha`
- `XRPUSDT` bid: `fixed_k_alpha`

## 5. How Many Features We Considered

In the main exhaustive single-feature search, we tested:

- `116` single-feature candidates per symbol-side pair

Those came from:

- last visible snapshot features
- 1-minute summaries
- 5-minute summaries
- 15-minute summaries
- 30-minute summaries
- 59-minute summaries

Important point:

- we tested **one feature at a time**
- we did **not** train a multi-feature joint model in the final deployed setup

## 6. Why We Only Used One Feature Per Adaptive Pair

This was mainly about robustness.

Given the number of training hours per pair, I thought a one-feature rule was the safest starting point because it is:

- easier to interpret
- easier to debug
- less likely to overfit than a richer joint model

So the tradeoff was:

- give up some model complexity
- gain clarity and a better chance of holding up out of sample

## 7. How We Went From Many Features To One

This happened in two stages.

### Stage A: search over candidate single-feature rules

For one pair at a time:

1. build the hour-level feature table from the first 59 minutes only
2. build the per-hour execution-cost table from `y_train`
3. for each candidate feature:
   - try `3`, `5`, and `7` bins
   - fit a bucketed rule
   - evaluate it by CV

The bucketed rule means:

- split the feature values into quantile bins
- learn the best schedule for each bin

### Stage B: nested CV picks the model class, not the exact feature

This part matters.

We found that the exact winning feature can move around across folds, especially when several candidates are near-tied.

So we used nested CV primarily to decide:

- which **model class** to trust for that pair

meaning:

- `fixed_k`
- `adaptive_k`
- `fixed_k_alpha`
- `adaptive_k_alpha`

Then, after freezing the class:

- re-run the single-feature search inside that class on full training data
- pick the final deployed feature + bin count

So the final deployed feature is:

- not “the one true feature forever”
- just the best full-train choice inside the already trusted class

## 8. How The Bins Were Chosen

For each candidate feature, we tried:

- `3` bins
- `5` bins
- `7` bins

The bins were formed by **quantiles** on the training subset.

That means:

- `3` bins = roughly bottom third / middle third / top third
- `5` bins = fifths
- `7` bins = sevenths

Important detail:

- each hour goes into **one** bin
- each bin has one learned execution rule attached to it

So the model is:

- feature value -> bin -> execution rule

## 9. How A Bin Gets Its Execution Rule

Once the hours are split into bins, we ask:

- among training hours in this bin, which action gives the lowest average implementation error?

If the class is `adaptive_k`, the action is:

- one `k` value

If the class is `adaptive_k_alpha`, the action is:

- one `(k, alpha)` schedule

So each bin stores:

- the best `k`, or
- the best `(k, alpha)`

This is learned from the offline cost table built using `y_train`.

## 10. Why The Feature Can Change Between 80% And 100%

This was one of the confusing parts, so here is the simple version.

When we run the frozen 80/20 holdout:

- the adaptive pair is re-fit inside the `80%` dev split
- so it can pick one single feature there

When we run the final deployment fit:

- the same class is re-fit on `100%` of training data
- so it can pick a different single feature

That does **not** mean the model suddenly uses multiple features.

It just means:

- the exact winning single-feature candidate can shift when the training sample changes

Sometimes this is harmless:

- several features are near-tied
- adding more data reshuffles the winner

Sometimes it is a warning sign:

- the 80% winner ranks very poorly in the full-train search

That is exactly why we used the 80/20 pass only as a veto/override layer for borderline pairs rather than as the final feature selector.

## 11. What The Holdout Override Logic Was

The holdout was **not** used to replace the deployed feature directly.

Instead:

- if the holdout looked bad and the adaptive feature choice was clearly unstable, we overrode the whole class to a simpler fixed model

That happened for:

- `ADAUSDT` bid
- `LTCUSDT` bid

For the other adaptive pairs, even if the exact feature changed between 80% and 100%, we still kept:

- the frozen model class from nested CV
- the final feature from the 100% full-train refit

## 12. The Main Point To Keep In Mind

The feature-selection process is best thought of like this:

- many candidate summaries of the first 59 minutes
- one of them is used to sort hours into bins
- each bin maps to one execution rule
- the whole thing stays small and interpretable

So the adaptive model is not magic.

It is basically:

- "use one summary of recent market state to decide which of a few schedule templates this hour should get"

That is the whole logic.
