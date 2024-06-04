import torch
from torch import nn
import torch.nn.functional as F

from .unet_parts import DownSample, ResidualBlock, UpSample, SwitchSequential
from .multi_head_attention import MultiHeadSpatialSelfAttention

class UNET(nn.Module):

    def __init__(self, in_channels: int, ch_init: int, ch_mult: list[int], attn_layers: list[int], d_model: int):
        """
            Backbone model for transforming images from one domain to another
        """

        super().__init__()
        self.head = nn.Conv2d(in_channels, ch_init, kernel_size=3, stride=1, padding=1)

        self.encoders = []
        ch_prev = ch_init
        for ind, mult in enumerate(ch_mult):
            self.encoders.append(SwitchSequential(
                ResidualBlock(ch_prev, ch_init * mult),
                MultiHeadSpatialSelfAttention(ch_init * mult, 8) if ind in attn_layers else nn.Identity()
            ))
            self.encoders.append(SwitchSequential(
                DownSample(ch_init * mult, ch_init * mult)
            ))
            ch_prev = ch_init * mult
        self.encoders = nn.ModuleList(self.encoders)

        self.bottle_neck = SwitchSequential(
            ResidualBlock(ch_prev, ch_prev),
            MultiHeadSpatialSelfAttention(ch_prev, ch_prev),
            ResidualBlock(ch_prev, ch_prev)
        )

        self.decoders = []
        for ind, mult in enumerate(reversed(ch_mult)):
            self.decoders.append(SwitchSequential(
                UpSample(ch_prev + ch_init * mult, ch_prev + ch_init * mult)
            ))
            self.decoders.append(SwitchSequential(
                ResidualBlock(ch_prev + ch_init * mult * 2, ch_init * mult),
                MultiHeadSpatialSelfAttention(ch_init * mult, 8) if (len(ch_mult) - ind - 1) in attn_layers else nn.Identity()
            ))
            ch_prev = ch_init * mult
        self.decoders = nn.ModuleList(self.decoders)

        self.tail = nn.Conv2d(ch_prev, 2 * in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Input Shape: (B, C, H, W)
            Output Shape: (B, 2 * C, H, W)
        """

        x = self.head(x)

        y = []
        for layer in self.encoders:
            x = layer(x)
            y.append(x.clone())
        
        x = self.bottle_neck(x)

        for layer in self.decoders:
            x = layer(torch.cat([x, y.pop()], dim=1))

        return self.tail(x)