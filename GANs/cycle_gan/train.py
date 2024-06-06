import torch
from torch import nn
import torchvision.transforms as transforms
from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os

from generator.unet import UNET
from discriminator.model import Discriminator

def train_gan(
    generator1,
    generator2,
    discriminator1,
    discriminator2,
    device,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 2e-4,
    betas: tuple = (0.5, 0.999),
    sample_dir: str = "samples",
):
    os.makedirs(sample_dir, exist_ok=True)

    generator1.to(device)
    generator2.to(device)
    discriminator1.to(device)
    discriminator2.to(device)

    # transforms
    transform = transforms.Compose([
        transforms.Resize(256, 256),
        transforms.HorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        transforms.ToTensor(),],
        additional_targets={"image0", "image"}
    )

    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ch_mult = [4, 16, 64, 256]
    attn_layers = [1, 3]
    generator1 = UNET(3, 4, ch_mult, attn_layers)
    generator2 = UNET(3, 4, ch_mult, attn_layers)

    discriminator1 = Discriminator(in_channels=3)
    discriminator2 = Discriminator(in_channels=3)

    train_gan(generator1, generator2, discriminator1, discriminator2, device)
