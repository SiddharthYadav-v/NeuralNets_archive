import torch
from torch import nn

from .attention import SpatialSelfAttention


class MultiHeadSpatialSelfAttention(nn.Module):
    
    def __init__(self, d_in : int, n_heads : int, in_bias : bool = False, out_bias : bool = False):
        """
            Computes spatial self attention for an image in different heads and combines them
        """
        
        assert d_in % n_heads == 0, "d_in must be a multiple of n_heads"
        
        super().__init__()
        self.heads = nn.ModuleList([
            SpatialSelfAttention(d_in, d_in // n_heads, bias = in_bias) for _ in range(n_heads)
        ])
        self.out = nn.Conv2d(d_in, d_in, kernel_size = 1, stride = 1, padding = 0, bias = out_bias)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
            Input Shape : (B, C, H, W)
            Output Shape : (B, C, H, W)
        """
        
        assert len(x.shape) == 4, "Input must be a batch of images in channels first mode"
        
        h = torch.cat([head(x) for head in self.heads], dim = 1)
        
        return self.out(h)