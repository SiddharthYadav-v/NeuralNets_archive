import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z):
        return self.double_conv(z)

class Generator(nn.Module):
    def __init__(self, codings_size):
        super().__init__()
        self.codings_size = codings_size

        self.generator = nn.Sequential(
            DoubleConv(1, 64, 32),
            nn.MaxPool2d(),
            DoubleConv(64, 256, 128),
            nn.MaxPool2d(),
            DoubleConv(256, 1024, 512),
            
        )