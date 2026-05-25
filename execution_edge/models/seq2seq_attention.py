"""Sequence-to-sequence model with additive attention.

Stacks a Bidirectional LSTM temporal encoder on top of the DeepLOB spatial
encoder, then decodes a 60-second mid-price trajectory autoregressively with
additive (Bahdanau-style) attention over the encoder output. Teacher forcing
is supported at training time via the ``y_teacher`` argument to ``forward``.
"""

import torch
import torch.nn as nn

from execution_edge.models.deeplob import DeepLOBEncoder


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over encoder outputs."""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W_e = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_d = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_outputs: torch.Tensor, dec_state: torch.Tensor) -> torch.Tensor:
        # enc_outputs: [B, T, E]; dec_state: [B, D]
        e = self.W_e(enc_outputs) + self.W_d(dec_state).unsqueeze(1)  # [B, T, A]
        scores = self.v(torch.tanh(e)).squeeze(-1)                     # [B, T]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * enc_outputs, dim=1)  # [B, E]
        return context


class Seq2SeqAttention(nn.Module):
    """DeepLOB encoder + BiLSTM + autoregressive attention decoder."""

    def __init__(
        self,
        hidden: int = 128,
        horizon: int = 60,
        dropout: float = 0.1,
        ask_bid_idx: tuple[int, int] = (0, 2),
    ):
        super().__init__()
        # Encoder: DeepLOB CNN, then BiLSTM over its per-timestep embeddings.
        self.spatial = DeepLOBEncoder(in_ch=1)
        self.encoder = nn.LSTM(
            input_size=192,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Decoder: additive attention + LSTM that consumes (prev_y, context).
        self.attn = AdditiveAttention(enc_dim=2 * hidden, dec_dim=hidden)
        self.decoder = nn.LSTM(
            input_size=1 + 2 * hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.out = nn.Linear(hidden, 1)
        self.init_h = nn.Linear(2 * hidden, hidden)
        self.init_c = nn.Linear(2 * hidden, hidden)
        self.horizon = horizon

        # Indices into the LOB feature vector where the level-1 ask and bid
        # prices live; used to seed the decoder with a mid-price proxy.
        self.ask_idx, self.bid_idx = ask_bid_idx

    def forward(self, x: torch.Tensor, y_teacher: torch.Tensor | None = None) -> torch.Tensor:
        """Predict the next ``horizon`` seconds of mid-price.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, T, F]`` LOB window.
        y_teacher : torch.Tensor, optional
            Shape ``[B, horizon]`` ground-truth mid-price trajectory, used for
            teacher forcing at training time. If ``None`` the decoder is
            fully autoregressive.
        """
        # Encode.
        h_spatial = self.spatial(x)             # [B, T, 192]
        enc_out, _ = self.encoder(h_spatial)    # [B, T, 2H]

        # Initialise decoder state from the encoder mean.
        enc_mean = enc_out.mean(dim=1)
        dec_h = torch.tanh(self.init_h(enc_mean)).unsqueeze(0)  # [1, B, H]
        dec_c = torch.tanh(self.init_c(enc_mean)).unsqueeze(0)

        # Seed the decoder with a mid-price proxy from the last encoder step.
        last_ask = x[:, -1, self.ask_idx]
        last_bid = x[:, -1, self.bid_idx]
        prev_y = ((last_ask + last_bid) / 2.0).unsqueeze(-1)  # [B, 1]

        outputs = []
        for t in range(self.horizon):
            context = self.attn(enc_out, dec_h[-1])           # [B, 2H]
            dec_in = torch.cat([prev_y, context], dim=1).unsqueeze(1)
            dec_out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            y_hat = self.out(dec_out).squeeze(1)              # [B, 1]
            outputs.append(y_hat.squeeze(-1))
            if self.training and y_teacher is not None:
                prev_y = y_teacher[:, t].unsqueeze(-1)
            else:
                prev_y = y_hat.detach()

        return torch.stack(outputs, dim=1)  # [B, horizon]
