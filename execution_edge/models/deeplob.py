"""DeepLOB CNN-Inception spatial encoder.

Processes the full (time, features) sequence of the visible limit-order-book
window as a 2D image, so that the convolution stack learns patterns across both
temporal context and book structure.

Architecture follows Tsantekidis et al. (2017) closely: three convolutional
blocks that progressively collapse the feature dimension while preserving
temporal length, followed by an Inception module with three parallel branches
(3-tick, 5-tick, and pooled) whose outputs are concatenated.

Input:  ``[B, T, 20]`` where the 20 features are five-level ask and bid
        prices and volumes interleaved.
Output: ``[B, T', 192]`` per-timestep feature embeddings ready to feed a
        recurrent decoder.
"""

import torch
import torch.nn as nn


class DeepLOBEncoder(nn.Module):
    """Three-block CNN with terminal Inception module for L2 order-book data."""

    def __init__(self, in_ch: int = 1):
        super().__init__()

        # Conv block 1: reduce features 20 -> 10 via a stride-2 convolution
        # over the feature dimension, then two temporal 4x1 convolutions.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # Conv block 2: reduce features 10 -> 5, two more temporal convolutions.
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        # Conv block 3: collapse remaining features 5 -> 1, two final temporal
        # convolutions. Output has feature dimension 1 and depth 32.
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # Inception module: three parallel branches operating at different
        # temporal receptive fields. Concatenating gives 64 + 64 + 64 = 192
        # channels per timestep.
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of LOB windows.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, T, F]`` where ``F`` is at least 20. Only the first 20
            features are consumed by the CNN; any extra features must be
            handled outside this module.

        Returns
        -------
        torch.Tensor
            Shape ``[B, T', 192]`` per-timestep feature embeddings.
        """
        # Use only the 20 LOB columns; reshape for 2D conv.
        x = x[:, :, :20]
        x = x.unsqueeze(1)  # [B, 1, T, 20]

        x = self.conv1(x)   # [B, 32, T', 10]
        x = self.conv2(x)   # [B, 32, T', 5]
        x = self.conv3(x)   # [B, 32, T', 1]

        # Inception: concat three branches along the channel dimension.
        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], dim=1)

        # Drop the singleton feature dim and put time before channels.
        x = x.squeeze(-1)              # [B, 192, T']
        x = x.permute(0, 2, 1)         # [B, T', 192]
        return x
