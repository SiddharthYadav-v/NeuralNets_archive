import torch
from torch import nn
import torch.nn.functional as F

from .unet_parts import DownSample, ResidualBlock, UpSample, SwitchSequential, TimeEmbedding
from .multi_head_attention import MultiHeadSpatialSelfAttention

class UNET(nn.Module):
    
    def __init__(self, in_channels: int, ch_init: int, ch_mult: list[int], attn_layers: list[int], time_dim: int, d_model: int, d_time: int):
        """
            Backbone model for the diffusion model, used to predict amount of noise added
        """

        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim, d_model, d_time)
        self.head = nn.Conv2d(in_channels, ch_init, kernel_size=3, stride=1, padding=1)

        self.encoders = []
        ch_prev = ch_init
        for ind, mult in enumerate(ch_mult):
            self.encoders.append(SwitchSequential(
                ResidualBlock(ch_prev, ch_init * mult, d_time),
                MultiHeadSpatialSelfAttention(ch_init * mult, 8) if ind in attn_layers else nn.Identity()
            ))
            self.encoders.append(SwitchSequential(
                DownSample(ch_init * mult, ch_init * mult)
            ))
            ch_prev = ch_init * mult
        self.encoders = nn.ModuleList(self.encoders)

        self.bottle_neck = SwitchSequential(
            ResidualBlock(ch_prev, ch_prev, d_time),
            MultiHeadSpatialSelfAttention(ch_prev, ch_prev),
            ResidualBlock(ch_prev, ch_prev, d_time)
        )

        self.decoders = []
        for ind, mult in enumerate(reversed(ch_mult)):
            self.decoders.append(SwitchSequential(
                UpSample(ch_prev + ch_init * mult, ch_prev + ch_init * mult)
            ))
            self.decoders.append(SwitchSequential(
                ResidualBlock(ch_prev + ch_init * mult * 2, ch_init * mult, d_time),
                MultiHeadSpatialSelfAttention(ch_init * mult, 8) if (len(ch_mult) - ind - 1) in attn_layers else nn.Identity()
            ))
            ch_prev = ch_init * mult
        self.decoders = nn.ModuleList(self.decoders)

        self.tail = nn.Conv2d(ch_prev, 2 * in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """
            Input Shape: (B, C, H, W) -> (B)
            Output Shape: (B, 2 * C, H, W)
        """

        t = self.time_embedding(t)
        x = self.head(x)

        y = []
        for layer in self.encoders:
            x = layer(x, t)
            y.append(x.clone())
        
        x = self.bottle_neck(x, t)

        for layer in self.decoders:
            x = layer(torch.cat([x, y.pop()], dim=1), t)

        return self.tail(x)
