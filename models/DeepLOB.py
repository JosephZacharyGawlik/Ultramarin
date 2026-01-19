"""
DeepLOB CNN-Inception encoder with proper temporal processing.

This version processes the full (Time, Features) sequence as a 2D "image",
allowing the CNN to learn patterns across both time and LOB structure.

Input:  [Batch, Time, 20]  (e.g., [32, 300, 20])
Output: [Batch, Time, 192] (sequence of feature vectors for LSTM)
"""

import torch
import torch.nn as nn


class DeepLOBEncoder(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        
        # Conv Block 1: Reduce features 20 → 10, temporal convolutions
        # Input: [B, 1, T, 20]
        self.conv1 = nn.Sequential(
            # Feature reduction: 20 → 10
            nn.Conv2d(in_ch, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            # Temporal conv: captures 4-second patterns
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            # Another temporal conv
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        
        # Conv Block 2: Reduce features 10 → 5, more temporal convolutions
        self.conv2 = nn.Sequential(
            # Feature reduction: 10 → 5
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            # Temporal conv
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            # Another temporal conv
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        
        # Conv Block 3: Collapse remaining features 5 → 1, final temporal convolutions
        self.conv3 = nn.Sequential(
            # Feature collapse: 5 → 1
            nn.Conv2d(32, 32, kernel_size=(1, 5)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            # Temporal conv
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            # Another temporal conv
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        
        # Inception Modules: Multi-scale temporal pattern extraction
        # Input to inception: [B, 32, T, 1]
        # These capture patterns at different time scales (1s, 3s, 5s)
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same'),  # 3-second patterns
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same'),  # 5-second patterns
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        # x: [Batch, Time, Features] e.g., [32, 300, 20]
        B, T, F = x.shape
        
        # 1. Standardize to 20 features and reshape for 2D CNN
        # [B, T, 20] → [B, 1, T, 20]
        x = x[:, :, :20]
        x = x.unsqueeze(1)  # Add channel dimension: [B, 1, T, 20]
        
        # 2. Convolutional Stages (with temporal interaction!)
        # The CNN now sees patterns across BOTH time and LOB structure
        x = self.conv1(x)  # [B, 32, T-2, 10]
        x = self.conv2(x)  # [B, 32, T-4, 5]
        x = self.conv3(x)  # [B, 32, T-6, 1]
        
        # 3. Inception Stage: Multi-scale temporal features
        # [B, 32, T', 1] → [B, 192, T', 1]
        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], dim=1)
        
        # 4. Reshape for LSTM: [B, 192, T', 1] → [B, T', 192]
        x = x.squeeze(-1)      # [B, 192, T']
        x = x.permute(0, 2, 1)  # [B, T', 192]
        
        return x
