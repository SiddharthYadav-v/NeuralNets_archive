import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, codings_size):
        super().__init__()
        self.codings_size = codings_size

        self.linear_1 = nn.Linear(self.codings_size, 128, bias = False)
        self.linear_2 = nn.Linear(128, 256, bias = False)
        self.linear_3 = nn.Linear(256, 512, bias = False)
        self.linear_4 = nn.Linear(512, 1024, bias = False)
        self.linear_5 = nn.Linear(1024, 28 * 28, bias = False)
        self.unflatten = nn.Unflatten(1, (28, 28))
    
    def forward(self, z):
        z = torch.relu(self.linear_1(z))
        z = torch.relu(self.linear_2(z))
        z = torch.relu(self.linear_3(z))
        z = torch.relu(self.linear_4(z))
        z = torch.tanh(self.linear_5(z))

        return self.unflatten(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(28 * 28, 256, bias = False)
        self.linear_2 = nn.Linear(256, 128, bias = False)
        self.linear_3 = nn.Linear(128, 64, bias = False)
        self.linear_4 = nn.Linear(64, 32, bias = False)
        self.linear_5 = nn.Linear(32, 1, bias = False)
    
    def forward(self, z):
        z = self.flatten(z)
        z = torch.relu(self.linear_1(z))
        z = torch.relu(self.linear_2(z))
        z = torch.relu(self.linear_3(z))
        z = torch.relu(self.linear_4(z))
        z = torch.sigmoid(self.linear_5(z))

        return z