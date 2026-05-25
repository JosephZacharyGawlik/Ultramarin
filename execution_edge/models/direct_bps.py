"""Direct-BPS model.

End-to-end execution model that consumes the visible limit-order-book window
and emits a 60-position schedule. Trained against a BPS-squared loss
propagated through the differentiable walk-the-book simulator in
``execution_edge.walk_the_book.differentiable_walk_the_book``.

Architecture:
    DeepLOB CNN encoder  ->  Bidirectional LSTM
                          ->  Multi-head cross-attention with 60 learned
                              query vectors over the per-timestep encoder
                              outputs
                          ->  per-second MLP head (hidden -> 1)
                          ->  add learnable logit bias (60 values, initialised
                              so that the softmax starts near a last-K TWAP)
                          ->  softmax over the 60 logits

The output is a probability distribution over the 60 seconds of the last
minute, multiplied by ``volume_to_fill`` outside the model to produce the
schedule. Approximately 722,000 trainable parameters.

The model is the basis for both the full direct-BPS experiment and the
60-parameter bias-only ablation reported in the discussion section: freezing
every parameter except ``logit_bias`` reduces the trainable parameter count
to 60 while leaving the inference path unchanged.
"""

import torch
import torch.nn as nn

from execution_edge.models.deeplob import DeepLOBEncoder


class DirectBPSModel(nn.Module):
    """DeepLOB + BiLSTM + cross-attention decoder + learnable TWAP-K bias."""

    def __init__(
        self,
        hidden: int = 128,
        dropout: float = 0.1,
        num_extra_features: int = 0,
        twap_k: int = 16,
    ):
        """
        Parameters
        ----------
        hidden : int
            Hidden dimension of the BiLSTM and the per-second MLP head.
        dropout : float
            Applied after the BiLSTM and inside the cross-attention layer.
        num_extra_features : int
            Number of non-LOB features per timestep (e.g. mid-price, OFI).
            These are concatenated to the DeepLOB embedding before the LSTM.
        twap_k : int
            ``logit_bias`` is initialised so that the first ``60 - twap_k``
            seconds have weight ``exp(-5)`` (about 0.007) and the last
            ``twap_k`` seconds have weight 1, producing a softmax that starts
            close to last-K TWAP. Set to the pair's dev-selected K.
        """
        super().__init__()
        self.num_extra_features = num_extra_features
        enc_dim = 2 * hidden  # bidirectional encoder

        # CNN-Inception spatial encoder over the LOB columns.
        self.spatial = DeepLOBEncoder(in_ch=1)

        # Per-timestep features go into a Bidirectional LSTM. Extra features
        # (mid-price, OFI, etc.) are concatenated to the DeepLOB embedding
        # before this step.
        encoder_input_size = 192 + num_extra_features
        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.enc_dropout = nn.Dropout(dropout)

        # Cross-attention decoder: 60 learned query vectors attend to the
        # encoder output, producing a per-second context vector of width
        # ``enc_dim``. Residual connection plus layer norm follow.
        self.queries = nn.Parameter(torch.randn(60, enc_dim) * 0.02)  # [60, 2H]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=enc_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(enc_dim)

        # Per-second MLP head: enc_dim -> hidden -> 1. The single output
        # becomes one of the 60 pre-softmax logits.
        self.head = nn.Sequential(
            nn.Linear(enc_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Learnable logit bias. Initialised to a near-binary pattern that
        # makes the initial softmax sit at last-K TWAP.
        self.logit_bias = nn.Parameter(torch.zeros(60))
        with torch.no_grad():
            self.logit_bias[: 60 - twap_k] = -5.0  # exp(-5) ~= 0.007

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a length-60 distribution over the last minute.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, T, F]``. The first 20 features must be the L2 LOB
            columns (the input the DeepLOB encoder consumes). Any additional
            features are concatenated to the DeepLOB embedding before the
            BiLSTM.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 60]``; rows are non-negative and sum to 1.
            Multiply by ``volume_to_fill`` outside the model to obtain the
            schedule that the simulator will execute.
        """
        # CNN over the 20 LOB columns.
        lob_features = x[:, :, :20]
        h_spatial = self.spatial(lob_features)        # [B, T', 192]

        # Concatenate any extra per-timestep features (mid-price, OFI, etc.).
        if self.num_extra_features > 0:
            T_prime = h_spatial.shape[1]
            extra = x[:, -T_prime:, 20:]              # [B, T', N_extra]
            h_spatial = torch.cat([h_spatial, extra], dim=-1)

        # Bidirectional LSTM.
        enc_out, _ = self.encoder(h_spatial)          # [B, T', 2H]
        enc_out = self.enc_dropout(enc_out)

        # Cross-attention: 60 query vectors attend to the encoder output.
        B = x.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, 60, 2H]
        decoded, _ = self.cross_attn(q, enc_out, enc_out)
        decoded = self.norm(decoded + q)                 # residual + LayerNorm

        # Per-second MLP head, then add the learnable bias and softmax.
        logits = self.head(decoded).squeeze(-1)          # [B, 60]
        logits = logits + self.logit_bias
        return torch.softmax(logits, dim=1)


def freeze_for_bias_only_ablation(model: DirectBPSModel) -> int:
    """Freeze every parameter except ``logit_bias``.

    Used for the bias-only ablation: the encoder and decoder become
    constants and only the 60 entries of the logit-bias vector are
    trained. Returns the number of remaining trainable parameters.
    """
    trainable = 0
    for name, param in model.named_parameters():
        if name == "logit_bias":
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    return trainable
