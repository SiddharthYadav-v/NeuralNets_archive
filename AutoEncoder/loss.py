import torch.nn as nn

def reconstruction_loss(x, x_hat):
    return nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')