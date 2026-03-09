"""
This replaces the simple Bi-LSTM head with a full autoregressive decoder.
"""

import torch
import torch.nn as nn
from models.DeepLOB import DeepLOBEncoder

class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W_e = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_d = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_outputs: torch.Tensor, dec_state: torch.Tensor) -> torch.Tensor:
        # enc_outputs: [B, T, E], dec_state: [B, D]
        e = self.W_e(enc_outputs) + self.W_d(dec_state).unsqueeze(1)  # [B, T, A]
        scores = self.v(torch.tanh(e)).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * enc_outputs, dim=1)  # [B, E]
        return context

class SuperModel(nn.Module):
    def __init__(self, hidden: int = 128, horizon: int = 60, dropout: float = 0.1,
                 ask_bid_idx=(0, 2), num_extra_features: int = 0):
        super().__init__()
        self.num_extra_features = num_extra_features

        # 1. Encoder Part
        self.spatial = DeepLOBEncoder(in_ch=1, dropout=dropout)
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

        # 2. Decoder Part
        self.attn = AdditiveAttention(enc_dim=2 * hidden, dec_dim=hidden)
        self.decoder = nn.LSTM(
            input_size=1 + 2 * hidden,  # previous y + context
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dec_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        self.init_h = nn.Linear(2 * hidden, hidden)
        self.init_c = nn.Linear(2 * hidden, hidden)
        self.horizon = horizon

        self.ask_idx, self.bid_idx = ask_bid_idx

    def forward(self, x: torch.Tensor, y_teacher: torch.Tensor = None,
                tf_ratio: float = 1.0) -> torch.Tensor:
        # x: [B, T, F] where F = 20 LOB + extra features
        # y_teacher: [B, 60]
        # tf_ratio: probability of using teacher forcing per step (1.0 = always, 0.0 = never)

        # Split LOB features (for CNN) and extra features (bypass CNN)
        lob_features = x[:, :, :20]
        h_spatial = self.spatial(lob_features)  # [B, T, 192]

        if self.num_extra_features > 0:
            extra_features = x[:, :, 20:]  # [B, T, N_extra]
            h_spatial = torch.cat([h_spatial, extra_features], dim=-1)  # [B, T, 192+N_extra]

        # Encode
        enc_out, _ = self.encoder(h_spatial)  # [B, T, 2H]
        enc_out = self.enc_dropout(enc_out)

        # Init decoder state from encoder mean
        enc_mean = enc_out.mean(dim=1)
        dec_h0 = torch.tanh(self.init_h(enc_mean))
        dec_c0 = torch.tanh(self.init_c(enc_mean))
        dec_h = dec_h0.unsqueeze(0)  # [1, B, H]
        dec_c = dec_c0.unsqueeze(0)

        outputs = []
        # Seed: use last encoder input midprice proxy (col 0 normalized)
        last_ask = x[:, -1, self.ask_idx]
        last_bid = x[:, -1, self.bid_idx]
        prev_y = ((last_ask + last_bid) / 2.0).unsqueeze(-1)  # [B, 1]

        for t in range(self.horizon):
            context = self.attn(enc_out, dec_h[-1])  # [B, 2H]
            dec_in = torch.cat([prev_y, context], dim=1).unsqueeze(1)  # [B, 1, 1+2H]
            dec_out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            dec_out = self.dec_dropout(dec_out)
            y_hat = self.out(dec_out).squeeze(1)  # [B, 1]
            outputs.append(y_hat.squeeze(-1))

            if self.training and y_teacher is not None:
                # Scheduled sampling: per-step stochastic teacher forcing
                if torch.rand(1).item() < tf_ratio:
                    prev_y = y_teacher[:, t].unsqueeze(-1)
                else:
                    prev_y = y_hat.detach()
            else:
                # Autoregressive at inference
                prev_y = y_hat.detach()

        return torch.stack(outputs, dim=1)  # [B, 60]
