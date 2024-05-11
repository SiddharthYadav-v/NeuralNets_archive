from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, noise):
        encoded = self.encoder(x, noise)
        decoded = self.decoder(encoded)

        return decoded