"""Execution Edge: HFT Challenge.

Code accompanying the report. Modules are organised by the role they play in
the evaluation pipeline:

    splits           Canonical SHA-1 hashed dev/holdout partition.
    preprocessing    LOB cleaning (DL stream) and last-minute frame
                     normalisation (adaptive stream).
    walk_the_book    Numpy reference simulator (re-exported from
                     ``data/simulate_walk_the_book.py``) and a differentiable
                     PyTorch variant used by the direct-BPS experiment.
    bps              Implementation-shortfall computation in basis points.
    schedules        Schedule constructors: TWAP family, predictive scheduler,
                     and the adaptive scheduling rule.
    features         Hour-level windowed summary features consumed by the
                     adaptive scheduling pipeline.
    candidates       Candidate-rule generation for the adaptive scheduling
                     tournament (fixed-k, fixed-(k,alpha), adaptive-k,
                     adaptive-(k,alpha)).
    selection        Nested cross-validation selection and the
                     holdout-as-veto pass.
    models           Neural-network models: DeepLOB encoder, BiLSTM with
                     attention decoder, and the direct-BPS model.
"""

__version__ = "1.0.0"
