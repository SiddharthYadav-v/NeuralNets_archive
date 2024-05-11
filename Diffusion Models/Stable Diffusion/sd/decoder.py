import torch
import torch.nn as nn
import torch.nn.functional as F
from sd.attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        # x: (bs, features, h, w)
        residue = x

        x = self.groupnorm(x)  # (bs, f, h, w) -> (bs, f, h, w)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))  # (bs, f, h, w) -> (bs, f, h * w)
        x = x.transpose(-1, -2)  # (bs, f, h * w) -> (bs, h * w, f)
        
        # Perform self-attention WITHOUT mask
        x = self.attention(x)  # (bs, h * w, f) -> (bs, h * w, f)
        x = x.transpose(-1, -2)  # (bs, h * w, f) -> (bs, f, h * w)
        x = x.view((n, c, h, w))  # (bs, f, h * w) -> (bs, f, h, w)
        x += residue  # (bs, f, h, w) + (bs, f, h, w) -> (bs, f, h, w)

        return residue  # (bs, f, w, h)

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
    
    def forward(self, x):
        # x: (bs, in_channels, h, w)

        residue = x

        x = self.groupnorm_1(x)  # (bs, in_channels, h, w) -> (bs, in_channels, h, w)
        x = F.silu(x)  # (bs, in_channels, h, w) -> (bs, in_channels, h, w)
        x = self.conv_1(x)  # (bs, in_channels, h, w) -> (bs, in_channels, h, w)
        x = self.groupnorm_2(x)  # (bs, in_channels, h, w) -> (bs, in_channels, h, w)
        x = F.silu(x)  # (bs, in_channels, h, w) -> (bs, in_channels, h, w)
        x = self.conv_2(x)  # (bs, in_channels, h, w) -> (bs, in_channels, h, w)

        return x + self.residual_layer(residue)  # (bs, in_channels, h, w) -> (bs, in_channels, h, w)
    
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),  # (bs, 4, h/8, w/8) -> (bs, 4, h/8, w/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),  # (bs, 512, h/8, w/8) -> (bs, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/8, w/8) -> (bs, 512, h/8, w/8)
            VAE_AttentionBlock(512),  # (bs, 512, h/8, w/8) -> (bs, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/8, w/8) -> (bs, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/8, w/8) -> (bs, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/8, w/8) -> (bs, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/8, w/8) -> (bs, 512, h/8, w/8)
            nn.Upsample(scale_factor=2),  # (bs, 512, h/8, w/8) -> (bs, 512, h/4, w/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (bs, 512, h/4, w/4) -> (bs, 512, h/4, w/4)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/4, w/4) -> (bs, 512, h/4, w/4)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/4, w/4) -> (bs, 512, h/4, w/4)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/4, w/4) -> (bs, 512, h/4, w/4)
            nn.Upsample(scale_factor=2),  # (bs, 512, h/4, w/4) -> (bs, 512, h/2, w/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (bs, 512, h/2, w/2) -> (bs, 512, h/2, w/2)
            VAE_ResidualBlock(512, 256),  # (bs, 512, h/2, w/2) -> (bs, 256, h/2, w/2)
            VAE_ResidualBlock(256, 256),  # (bs, 256, h/2, w/2) -> (bs, 256, h/2, w/2)
            VAE_ResidualBlock(256, 256),  # (bs, 256, h/2, w/2) -> (bs, 256, h/2, w/2)
            nn.Upsample(scale_factor=2),  # (bs, 256, h/2, w/2) -> (bs, 256, h, w)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), VAE_ResidualBlock(256, 256),  # (bs, 256, h, w) -> (bs, 256, h, w)
            VAE_ResidualBlock(256, 128),  # (bs, 256, h, w) -> (bs, 128, h, w)
            VAE_ResidualBlock(128, 128),  # (bs, 128, h, w) -> (bs, 128, h, w)
            nn.GroupNorm(32, 128),  # (bs, 128, h, w) -> (bs, 128, h, w)
            nn.SiLU(),  # (bs, 128, h, w) -> (bs, 128, h, w)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # x: (bs, 4, h/8, w/8)

        x /= 0.18215

        for module in self:
            x = module(x)
        
        # (bs, 3, h, w)
        return x