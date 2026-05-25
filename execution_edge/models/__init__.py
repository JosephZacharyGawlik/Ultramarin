"""Neural-network models used by the deep-learning pipeline and the
direct-BPS optimisation experiment.

Modules:
    deeplob              DeepLOB CNN-Inception spatial encoder.
    seq2seq_attention    Bidirectional LSTM temporal encoder with additive
                         attention decoder for autoregressive mid-price prediction.
    direct_bps           End-to-end model that consumes LOB features and emits
                         a 60-second execution schedule, trained against BPS-squared
                         through a differentiable walk-the-book simulator.
"""
