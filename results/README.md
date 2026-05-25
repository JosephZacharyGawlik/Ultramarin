# `results/`

Output artifacts produced by the notebooks land here. The canonical
contents after a full run are:

- `holdout_bps_per_pair.csv` headline per-pair BPS for every approach
  (TWAP-K, predictive scheduler, adaptive scheduling, direct-BPS), produced
  by `notebooks/05_holdout_evaluation.ipynb`.
- `deployed_rules.md` per-pair deployed adaptive scheduling rules
  (ask and bid sides), produced by
  `notebooks/03_adaptive_scheduling.ipynb`.

This folder is intentionally empty in the shipped repository. Running the
notebooks repopulates it.
