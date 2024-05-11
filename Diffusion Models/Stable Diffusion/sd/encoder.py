import torch
import torch.nn as nn
import torch.nn.functional as F
from sd.decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # (bs, c, h, w) -> (bs, 128, h, w)
            VAE_ResidualBlock(128, 128),  # (bs, 128, h, w) -> (bs, 128, h, w)
            VAE_ResidualBlock(128, 128),  # (bs, 128, h, w) -> (bs, 128, h,  w)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),  # (bs, 128, h, w) -> (bs, 128, h / 2, w / 2)
            VAE_ResidualBlock(128, 256),  # (bs, 128, h / 2, w / 2) -> (bs, 256, h / 2,  w / 2)
            VAE_ResidualBlock(256, 256),  # (bs, 256, h / 2, w / 2) -> (bs, 256, h / 2, w / 2)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),  # (bs, 256, h / 2, w / 2) -> (bs, 256, h / 4, w / 4)
            VAE_ResidualBlock(256, 512),  # (bs, 256, h / 4, w / 4) -> (bs, 512, h / 4, w / 4)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h / 4, w / 4) -> (bs, 512, h / 4, w / 4)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),  # (bs, 512, h / 4, w / 4) -> (bs, 512, h / 8, w / 8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h / 8, w / 8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h / 8, w / 8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h / 8, w / 8)
            VAE_AttentionBlock(512),  # (bs, 512, h / 8, w / 8)
            VAE_ResidualBlock(512, 512),  # (bs, 512, h / 8, w / 8)
            nn.GroupNorm(32, 512),  # (bs, 512, h / 8, w / 8)
            nn.SiLU(),  # (bs, 512, h / 8, w / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        # x: (bs, c, h, w)
        # noise: (bs, 4, h / 8, w / 8)
        for module in self:

            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        
        mean, log_variance = torch.chunk(x, 2, dim=1)  # (bs, 8, h/8, w/8) -> two tensors of shape (bs, 4, h/8, w/8)
        log_variance = torch.clamp(log_variance, -30, 20)  # (bs, 4, h/8, w/8) -> (bs, 4, h/8, w/8)
        variance = log_variance.exp() # (bs, 4, h/8, w/8) -> (bs, 4, h/8, w/8)
        stdev = variance.sqrt()  # (bs, 4, h/8, w/8) -> (bs, 4, h/8, w/8)

        # Transform N(0, 1) -> N(mean, stdev)
        # (bs, 4, h/8, w/8) -> (bs, 4, h/8, w/8)
        x = mean + stdev * noise

        x *= 0.18215

        return x