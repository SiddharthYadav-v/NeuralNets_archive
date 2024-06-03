import torch
from torch import nn

class Diffusion(nn.Module):

    def __init__(self, time_steps: int):
        """
            Computes the forward process for diffusion
        """

        super().__init__()
        time_steps_vec = torch.tensor([torch.pi / 2 * ((t / time_steps + 0.008) / 1.008) for t in range(0, time_steps + 1)])
        f = torch.cos(time_steps_vec).square()
        alpha_bar = f / f[0]
        beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        beta = beta.clamp(max = 0.999)
        beta_bar = (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:]) * beta
        alpha_bar = alpha_bar[1:]
        alpha = 1.0 - beta

        self.register_buffer("beta", beta)
        self.register_buffer("beta_bar", beta_bar)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("one_minus_alpha_bar", (1.0 - alpha_bar))
        self.register_buffer("sqrt_beta", beta.sqrt())
        self.register_buffer("sqrt_alpha", alpha.sqrt())
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

    def forward(self, x: torch.Tensor, noise: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """
            Input Shape: (B, C, H, W), (B, C, H, W), (B)
            Output Shape: (B, C, H, W)
        """

        assert x.shape == noise.shape, "Image input and noise input shape must be same"

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1).to(x.device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1).to(x.device)

        return sqrt_alpha_bar_t * x + sqrt_alpha_bar_t * noise
