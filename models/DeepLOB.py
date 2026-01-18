
"""
This is the standard DeepLOB CNN-Inception encoder.
"""

import torch
import torch.nn as nn

class DeepLOBEncoder(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        # Conv Blocks 1, 2, 3 (Keep these exactly as you have them)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=(1,2), stride=(1,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,kernel_size=(3,1), padding=(1,0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,kernel_size=(3,1), padding=(1,0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,2),stride=(1,1)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,kernel_size=(3,1), padding=(1,0)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,kernel_size=(3,1), padding=(1,0)), nn.Tanh(), nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,kernel_size=(3,1), padding=(1,0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,kernel_size=(3,1), padding=(1,0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        # Inception Modules (Keep these exactly as you have them)
        self.inp1 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=(1,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,kernel_size=(3,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=(1,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,kernel_size=(5,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3,1),stride=(1,1),padding=(1,0)),
            nn.Conv2d(32,64,kernel_size=(1,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )

    def forward(self, x):
        # x: [Batch, Time, Features]
        B, T, F = x.shape
        
        # 1. Standardize to 20 features (5 levels * 4)
        x = x[:, :, :20] 
        
        # 2. Reshape for CNN: [B*T, 1, 5, 4]
        x = x.reshape(B * T, 1, 5, 4) 
        
        # 3. Convolutional Stages
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 4. Inception Stage
        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], dim=1) # [B*T, 192, 5, 1]
        
        # 5. Global Average Pooling (The Fix)
        # This collapses the '5' (height) into 1, giving us [B*T, 192, 1, 1]
        x = torch.mean(x, dim=(2, 3)) 
        
        # 6. Final Shape for LSTM: [Batch, Time, 192]
        return x.view(B, T, 192)