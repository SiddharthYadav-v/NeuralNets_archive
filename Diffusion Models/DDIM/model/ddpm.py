import torch
from torch import nn

from .diffusion import Diffusion
from .unet import UNET

class DDPM(nn.Module):

    def __init__(self, in_channels: int, ch_init: int, ch_mult: list[int], attn_layers: list[int], time_dim: int, d_model: int, d_time: int):
        """
            Computes forward process noisy image and returns predicted noise as well as log variance from backward process
        """

        super().__init__()
        self.time_steps = time_dim
        self.diffusion = Diffusion(time_dim)
        self.unet = UNET(in_channels, ch_init, ch_mult, attn_layers, time_dim, d_model, d_time)

    def forward(self, x: torch.Tensor, noise: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """
            Input Shape: (B, C, H, W), (B, C, H, W), (B)
            Output Shape: (B, 2 * C, H, W)
        """

        assert x.shape == noise.shape, "Image input and noise input shape must be same"
        
        x_t = self.diffusion(x, noise, t)
        eps, v = torch.chunk(self.unet(x_t, t), 2, dim = 1)

        return eps, v
    
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
            Input Shape: (B, C, H, W)
            Output Shape: (B, C, H, W)
        """

        b, c, h, w = x.shape
        for t in range(self.time_steps, 0, -1):
            z = torch.randn(b, c, h, w).to(x.device) if t > 1 else torch.zeros(b, c, h, w).to(x.device)
            time_tensor = torch.tensor([t - 1] * b).long().to(x.device)
            eps, v = torch.chunk(self.unet(x, time_tensor), 2, dim = 1)

            sqrt_alpha_t = self.diffusion.sqrt_alpha[t - 1]
            beta_t = self.diffusion.beta[t - 1]
            beta_bar_t = self.diffusion.beta_bar[t - 1]
            sqrt_one_minus_alpha_bar_t = self.diffusion.sqrt_one_minus_alpha_bar[t - 1]

            x = 1 / sqrt_alpha_t * (x - beta_t / sqrt_one_minus_alpha_bar_t * eps) + torch.exp(v * torch.log(beta_t) + (1.0 - v) * torch.log(beta_bar_t)) * z

        return x
