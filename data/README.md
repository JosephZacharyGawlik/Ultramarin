# `data/` layout

This folder holds the per-pair training data provided by the practical
organisers. The parquet files are not redistributed in this repository.

Expected layout (seven symbols):

```
data/
├── simulate_walk_the_book.py    reference execution simulator (shipped)
├── BTCUSDT/
│   ├── X_train.parquet          first 59 minutes of L2 LOB data per hour
│   ├── y_train.parquet          last minute of L2 LOB data + close + volume
│   ├── X_test.parquet           held-out test split (used for the submission)
│   └── vol_to_fill.txt          per-pair target volume (e.g. "4.0")
├── ETHUSDT/
├── LTCUSDT/
├── SOLUSDT/
├── ADAUSDT/
├── DOGEUSDT/
└── XRPUSDT/
```

## Schema

Every parquet file has one row per `(anonymized_id, time_in_hour)` pair:

- `anonymized_id`        unique hour identifier (`uint64`), shared across
                         pairs when the same wall-clock hour appears in
                         multiple pairs.
- `time_in_hour`         offset within the hour as a `Timedelta`.
- `ask_price_1..5`       five-level ask prices.
- `ask_vol_1..5`         five-level ask volumes.
- `bid_price_1..5`       five-level bid prices.
- `bid_vol_1..5`         five-level bid volumes.
- `open`, `high`, `low`, `close`  OHLC trade summary for the second.
- `volume`               OHLCV trade volume for the second.

`X_train.parquet` covers `time_in_hour` from `0` to `58:59`; `y_train.parquet`
covers `59:00` to `59:59`.

## Per-pair counts

| Pair  | Hours (sessions) |
|-------|-----------------|
| BTC   | 705             |
| ETH   | 705             |
| LTC   | 704             |
| SOL   | 705             |
| ADA   | 695             |
| DOGE  | 703             |
| XRP   | 705             |

Pooled unique `anonymized_id` values across the seven pairs: 1,008. The
SHA-1 holdout used throughout the report contains 202 of these IDs.
