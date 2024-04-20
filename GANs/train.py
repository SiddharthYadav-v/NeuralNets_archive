import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import os

from gan import gan

def train_gan(
    generator,
    discriminator,
    device,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 2e-4,
    betas: tuple = (0.5, 0.999),
    sample_dir: str = "samples",
):
    os.makedirs(sample_dir, exist_ok=True)

    generator.to(device=device)
    discriminator.to(device=device)

    adversarial_loss = nn.BCELoss()
    adversarial_loss.to(device=device)

    # transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # dataset and dataloader                           transform=transform)
    dataset = datasets.MNIST(root='./data', download=True, train=True,
                             transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, drop_last=True)

    # optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas = betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas = betas)
    fixed_noise = torch.randn((25, codings_size, )).to(device)

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            real_images, _ = data
            real_images = real_images.to(device)

            # Train discriminator
            optimizer_D.zero_grad()

            # Real images
            real_labels = torch.ones(batch_size, 1).to(device)
            real_outputs = discriminator(real_images).view(-1, 1)
            d_loss_real = adversarial_loss(real_outputs, real_labels)

            # Fake images
            noise = torch.randn(batch_size, codings_size).to(device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_outputs = discriminator(fake_images.detach()).view(-1, 1)
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

            # Backpropagation
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_images).view(-1, 1)
            g_loss = adversarial_loss(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Print loss
            if i % 100 == 0:
                print(
                    f"Epoch: {epoch}/{epochs}, Batch: {i}/{len(dataloader)} "
                    f"D loss: {d_loss.item()}, G loss: {g_loss.item()}"
                )
        
        with torch.no_grad():
            generated_images = generator(fixed_noise)
            save_image(generated_images.unsqueeze(1), os.path.join(sample_dir, f"epoch_{epoch}.png"), nrow=5, normalize=True)

if __name__ == '__main__':
    codings_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = gan.Generator(codings_size=codings_size)
    discriminator = gan.Discriminator()

    train_gan(generator, discriminator, device, sample_dir='vanilla-gan')
