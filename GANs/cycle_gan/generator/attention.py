import torch
from torch import nn
import torch.nn.functional as F

import math

class SpatialSelfAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, bias: bool=False):
        """
            Computes Spatial Self Attention for an image
        """

        super().__init__()
        self.query = nn.Conv2d(d_in, d_out, kernel_size=1, stride=1, padding=0, bias=bias)
        self.key = nn.Conv2d(d_in, d_out, kernel_size=1, stride=1, padding=0, bias=bias)
        self.value = nn.Conv2d(d_in, d_out, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Input Shape: (B, C, H, W)
            Output Shape: (B, C, H, W)
        """

        assert len(x.shape) == 4, "Input must be a batch of images in channel first mode"

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        b, c, h, w = q.shape

        q = q.view(b, c, h * w).transpose(-1, -2)
        k = k.view(b, c, h * w).transpose(-1, -2)
        v = v.view(b, c, h * w).transpose(-1, -2)

        wei = q @ k.transpose(1, 2) / math.sqrt(c)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v

        out = out.transpose(1, 2).view(b, c, h, w)

        return out