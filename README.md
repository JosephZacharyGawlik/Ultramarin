# Execution Edge: HFT Challenge

Code accompanying the final report for the LMU Munich Data Science Practical
(WS 25/26) Execution Edge challenge.

The challenge asks, for each one-hour trading session across seven
USDT-quoted cryptocurrency pairs, for a fixed split of a per-pair buy order
(and a fixed split of a per-pair sell order) across the final 60 seconds of
the hour. The split is decided in advance from the first 59 minutes of L2
limit-order-book data, and the schedule is scored in basis points by the
implementation shortfall against the final close price, with a volume-fill
penalty applied to underfilled hours.

## Repository layout

```
.
├── data/
│   ├── simulate_walk_the_book.py     reference execution simulator (provided)
│   └── <SYMBOL>/                     per-pair parquet files (user-supplied)
│       ├── X_train.parquet
│       ├── y_train.parquet
│       ├── X_test.parquet
│       └── vol_to_fill.txt
│
├── execution_edge/                   the Python package consumed by notebooks
│   ├── splits.py                     SHA-1 hashed dev/holdout partition
│   ├── preprocessing.py              LOB cleaning + last-minute reindexing
│   ├── walk_the_book.py              numpy reference + differentiable variant
│   ├── bps.py                        implementation shortfall metric
│   ├── data.py                       column-name and time-index constants,
│   │                                 parquet and vol_to_fill loaders
│   ├── schedules.py                  TWAP-K, TWAP-K-alpha, capping overlay
│   ├── predictive_scheduler.py       deep-learning pipeline's scheduler
│   ├── features.py                   hour-level summary features
│   ├── candidates.py                 candidate feature/bin pools
│   ├── selection.py                  nested cross-validation selection
│   ├── evaluation.py                 holdout scoring helpers
│   └── models/
│       ├── deeplob.py                CNN-Inception spatial encoder
│       ├── seq2seq_attention.py      BiLSTM + attention decoder
│       └── direct_bps.py             direct-BPS end-to-end model
│
├── notebooks/
│   ├── 01_baselines.ipynb            TWAP family and per-pair K selection
│   ├── 02_deep_learning.ipynb        DeepLOB + BiLSTM + predictive scheduler
│   ├── 03_adaptive_scheduling.ipynb  binning + capping + nested CV
│   ├── 04_direct_bps.ipynb           direct-BPS + bias-only ablation
│   └── 05_holdout_evaluation.ipynb   canonical SHA-1 holdout BPS table
│
├── results/                          notebook outputs and saved schedules
├── requirements.txt
└── README.md
```

## Section map to the report

| Report section                              | Notebook                           | Modules used                                          |
|---------------------------------------------|------------------------------------|-------------------------------------------------------|
| 6.2  TWAP family                            | `01_baselines.ipynb`               | `splits`, `schedules`, `predictive_scheduler`         |
| 6.3  Deep-learning pipeline                 | `02_deep_learning.ipynb`           | `preprocessing`, `models.seq2seq_attention`, `predictive_scheduler` |
| 6.4  Adaptive scheduling                    | `03_adaptive_scheduling.ipynb`     | `features`, `candidates`, `schedules`, `selection`    |
| 8.1  Direct-BPS optimisation                | `04_direct_bps.ipynb`              | `models.direct_bps`, `walk_the_book`, `bps`           |
| Headline holdout results (Sec. 6.3 and 6.4) | `05_holdout_evaluation.ipynb`      | every approach above                                   |

## Setup

The notebooks expect Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the seven per-pair parquet files supplied with the challenge into
`data/<SYMBOL>/` so the layout looks like:

```
data/BTCUSDT/X_train.parquet
data/BTCUSDT/y_train.parquet
data/BTCUSDT/X_test.parquet
data/BTCUSDT/vol_to_fill.txt
data/ETHUSDT/...
... (seven symbols total)
```

The parquet files are not redistributed in this repository; they come from
the practical organisers.

## How to reproduce the report's numbers

The fastest route is `notebooks/05_holdout_evaluation.ipynb`, which scores
every approach on the canonical SHA-1 holdout and produces the headline
per-pair BPS table that appears in Sections 6.3 and 6.4 of the report. The
holdout evaluation runs on CPU in about ten minutes once the deep-learning
models have been trained.

The deep-learning pipeline of Section 6.3 and the direct-BPS experiment of
Section 8.1 require GPU training. Each per-pair run is roughly 30 to 60
minutes of GPU time on a Colab T4. Training the seven pairs end to end is
therefore the slowest path in this repository; everything else is CPU.

The adaptive scheduling pipeline of Section 6.4 runs entirely on CPU. The
full nested-CV tournament across all candidate features, bin counts, and
capping overlays takes a few hours on a laptop.

## Evaluation conventions

Every BPS number in the report comes from the same evaluation pipeline:

1. **Holdout split.** SHA-1 hash each `anonymized_id` and take the top 20 %
   of the pooled IDs as the holdout. The remaining 80 % are the dev
   partition. The split is global; each pair's holdout is the intersection
   of these 202 IDs with that pair's training data. See
   `execution_edge/splits.py`.

2. **Per-pair K.** Sweep K = 1, ..., 60 on the dev partition only and pick
   the K that minimises mean TWAP-K BPS. The deep-learning pipeline and the
   adaptive scheduling pipeline both use this same dev-selected K. See the
   sweep helper in `notebooks/01_baselines.ipynb`.

3. **Last-minute handling.** Every holdout hour's last-minute frame is
   reindexed to the canonical 60-second grid (seconds 59:00 through 59:59);
   missing seconds become NaN-filled rows that the simulator handles
   gracefully. See `execution_edge/preprocessing.normalize_last_minute_frame`.

4. **BPS formula.**
   ```
   impl_error = |vwap - close| / close * 10_000
   penalty    = min(100, volume_to_fill / volume_executed)
   bps        = impl_error * penalty
   ```
   The per-pair report number is the mean of this value across that pair's
   holdout hours. See `execution_edge/bps.compute_bps`.

5. **Simulator.** The reference numpy simulator in
   `data/simulate_walk_the_book.py` is used everywhere in the repository
   (including by the differentiable PyTorch reimplementation, which has been
   verified to agree to floating-point precision).
