import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import optim
import matplotlib.pyplot as plt

from vae import *

data_dir = "E:\ClassyPics\Animals10"
dataset = torchvision.datasets.ImageFolder(data_dir,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5,),
                                                                    (0.5, 0.5, 0.5))]))

def train_vae(
    model,
    device,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 2e-4,
    betas: tuple = (0.5, 0.999),
    shape: tuple = (256, 256),
):  
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (img, _) in enumerate(dataloader):
            img = img.to(device)
            noise = torch.randn((4, shape[0] // 8, shape[1] // 8)).to(device)

            optimizer.zero_grad()
            x_hat = model(img, noise)
            loss = F.binary_cross_entropy(x_hat, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}/{i + 1}], loss: {running_loss / 200}')
                running_loss = 0.0

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()
    model = VAE(encoder, decoder)
    shape = (256, 256)
    transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))])
    
    data_dir = "E:\ClassyPics\Animals10"
    batch_size = 64
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, drop_last=True)

    train_vae(model, device)

    dataiter = iter(dataloader)
    
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_hat = model(x)

            break

    def show_image(x, idx):
        x = x.view(batch_size, 256, 256)

        fig = plt.figure()
        plt.imshow(x[idx].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()
    
    show_image(x, idx=0)
    show_image(x, idx=0)
