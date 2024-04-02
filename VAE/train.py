import torch
import torch.nn as nn
import torch.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

from model import *
from loss import loss_function

def train_model(
    model,
    device,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    x_dim = 28 * 28
    # 1. Load Dataset
    train_set = MNIST(root='/data', download=True, train=True,
                      transform=transform)
    # 2. Create Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    # 3. Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training loop
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)

            # zero the parameter gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch + 1}, avg_loss: {epoch_loss / (idx * batch_size)}")
    print("Finished training")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200
    epochs = 30
    lr = 1e-3

    encoder = Encoder(x_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, x_dim)
    model = Model(encoder, decoder).to(device)

    train_model(model, device, epochs=epochs, lr=lr)

    # Save the trained model
    PATH = 'VAE/MNIST_vae.pth'
    torch.save(model.state_dict(), PATH)
