import torch
from torch import nn
import torch.nn.functional as F

import math

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        """
            [GroupNorm => SiLU => Conv2d] * 2
        """
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Input Shape: (B, C_in, H, W)
            Output Shape: (B, C_out, H, W)
        """

        residue = x
        x = self.block1(x)
        x = self.block2(x)

        return x + self.residual_layer(residue)

class DownSample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """
            Halves the dimensions of the input image using a convolutional layer
        """

        super().__init__()
        self.layer = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Input Shape: (B, C_in, H, W)
            Output Shape: (B, C_out, H // 2, W // 2)
        """

        return self.layer(x)

class UpSample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """
            Doubles the dimensions of input image and then passes through a convolutional layer
        """

        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Input Shape: (B, C_in, H, W)
            Output Shape: (B, C_out, H * 2, W * 2)
        """

        return self.layer(x)
    
class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
            Input Shape: (B, C_in, H, W)
            Output Shape: (B, C_out, H, W)
        """

        for layer in self:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            
        return x