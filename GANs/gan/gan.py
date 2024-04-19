import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, codings_size):
        super().__init__()

        self.linear_1 = nn.Linear(codings_size, 128)
        self.linear_2 = nn.Linear(128, 256)
        self.linear_3 = nn.Linear(256, 512)
        self.linear_4 = nn.Linear(512, 1024)
        # self.linear_5 = nn.Linear(1024, 3 * 1024)
        self.linear_5 = nn.Linear(1024, 28 * 28)
        # self.unflatten = nn.Unflatten(1, (3, 32, 32))
        self.unflatten = nn.Unflatten(1, (28, 28))
    
    def forward(self, z):
        z = torch.relu(self.linear_1(z))
        z = torch.relu(self.linear_2(z))
        z = torch.relu(self.linear_3(z))
        z = torch.relu(self.linear_4(z))
        z = torch.sigmoid(self.linear_5(z))

        return self.unflatten(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        # self.linear_1 = nn.Linear(3 * 1024, 1024)
        self.linear_1 = nn.Linear(28 * 28, 1024)
        self.linear_2 = nn.Linear(1024, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 128)
        self.linear_5 = nn.Linear(128, 1)
    
    def forward(self, z):
        z = self.flatten(z)
        z = torch.relu(self.linear_1(z))
        z = torch.relu(self.linear_2(z))
        z = torch.relu(self.linear_3(z))
        z = torch.relu(self.linear_4(z))
        z = torch.sigmoid(self.linear_5(z))

        return z