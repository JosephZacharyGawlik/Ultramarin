"""Multi-target forecasting model.

DeepLOB CNN-Inception encoder followed by a Bidirectional LSTM and a single
linear head that emits a length-60 trajectory for each of three targets
simultaneously: mid-price, spread, and top-of-book liquidity. Used by the
volume and spread experiments documented in Section 9 of the report.

The architecture is non-autoregressive: the encoder summarises the visible
59-minute window and the head predicts all 60 seconds of all three targets
in one forward pass.
"""

import torch
import torch.nn as nn

from execution_edge.models.deeplob import DeepLOBEncoder


TARGET_NAMES = ("mid", "spread", "liq")


class DeepLOBForecastMulti(nn.Module):
    """DeepLOB + BiLSTM + linear head predicting three trajectories at once."""

    def __init__(self, embed_dim: int = 192, hidden: int = 128,
                 horizon: int = 60, heads: int = len(TARGET_NAMES)):
        super().__init__()
        self.encoder_cnn = DeepLOBEncoder(in_ch=1)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, horizon * heads)
        self.horizon = horizon
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape ``[B, T, 20]``
            LOB window. Only the 20 LOB columns are consumed.

        Returns
        -------
        torch.Tensor, shape ``[B, horizon, heads]``
            Per-second predictions for each target.
        """
        feats = self.encoder_cnn(x)
        h0 = torch.zeros(2, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(2, x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(feats, (h0, c0))
        last = out[:, -1, :]
        flat = self.head(last)
        return flat.view(x.size(0), self.horizon, self.heads)


def multi_head_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    head_weights: tuple[float, float, float] = (1.0, 0.2, 0.2),
    smooth_lambda: float = 0.02,
) -> torch.Tensor:
    """MSE across the three heads, weighted, plus a smoothness penalty on mid.

    ``pred`` and ``target`` are both shape ``[B, horizon, 3]``. The smoothness
    penalty is the squared first-difference of the mid-price head, summed and
    multiplied by ``smooth_lambda``. The default head weights up-weight mid
    relative to spread and liquidity.
    """
    weights = torch.tensor(head_weights, device=pred.device, dtype=pred.dtype)
    mse = ((pred - target) ** 2) * weights
    loss = mse.mean()
    if smooth_lambda > 0 and pred.shape[-1] > 0:
        mid = pred[:, :, 0]
        smooth = (mid[:, 1:] - mid[:, :-1]) ** 2
        loss = loss + smooth_lambda * smooth.mean()
    return loss
