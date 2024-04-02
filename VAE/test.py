import torch
from model import *
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_dim = 784
hidden_dim = 400
latent_dim = 200
epochs = 10
lr = 1e-3
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
])

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = Model(encoder, decoder).to(device)

PATH = 'VAE/MNiST_vae.pth'

model.load_state_dict(torch.load(PATH))

test_set = MNIST(root='/data', download=True, train=False,
                 transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=True, drop_last=True)

model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(device)

        x_hat, _, _ = model(x)

        break

def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

show_image(x, idx=0)
show_image(x_hat, idx=0)
