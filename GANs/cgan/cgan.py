import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, codings_size, embeddings_size):
        super().__init__()
        self.codings_size = codings_size
        self.embeddings_size = embeddings_size

        self.label_emb = nn.Embedding(10, 10)

        self.generate = nn.Sequential(
            nn.Linear(codings_size + embeddings_size, 7 * 7 * 128),  # (codings_size + embeddings_size) -> (7 * 7 * 128)
            nn.Unflatten(1, (128, 7, 7)),  # (128 * 7 * 7) -> (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # (128, 7, 7) -> (128, 14, 14)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # (128, 14, 14) -> (128, 14, 14)
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # (128, 14, 14) -> (128, 28, 28)
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # (128, 28, 28) -> (64, 28, 28)
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),  # (64, 28, 28) -> (1, 28, 28)
            nn.Tanh(),
        )
    
    def forward(self, z, labels):
        z = torch.cat((self.label_emb(labels), z), -1)
        return self.generate(z)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_embed = nn.Sequential(
            nn.Embedding(10, 10),
            nn.Linear(10, 14 * 14),
            nn.Linear(14 * 14, 1 * 28 * 28),
            nn.Unflatten(1, (1, 28, 28)),
        )

        self.discriminate = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),  # (1, 28, 28) -> (16, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (16, 14, 14) -> (32, 7, 7)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (32, 7, 7) -> (64, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            nn.Conv2d(64, 128, 3, 2, 1),  # (64, 4, 4) -> (128, 2, 2)
            nn.Flatten(),
            nn.Linear(2 * 2 * 128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, z, labels):
        label_embedding = self.label_embed(labels)
        return self.discriminate(torch.cat([label_embedding, z], dim=1))