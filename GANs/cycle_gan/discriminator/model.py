import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.project = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1)

        self.discriminate = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (3, 256, 256) => (16, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (16, 128, 128) => (32, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, 64, 64) => (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # (64, 32, 32) => (64, 16, 16)
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # (64, 16, 16) => (64, 8, 8)
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            nn.Flatten(),  # (64, 8, 8) => (64 * 8 * 8)
            nn.Linear(64 * 8 * 8, 32 * 2 * 2, bias=False),  # (64 * 8 * 8) => (32 * 2 * 2)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32 * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.project(z)
        return self.discriminate(z)