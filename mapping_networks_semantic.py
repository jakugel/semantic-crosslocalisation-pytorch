import torch.nn as nn
import torch
from utility_layers import EqualizedLinear


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers, equalized=True, map_leaky=False):
        super().__init__()

        if not map_leaky:
            act = nn.LeakyReLU(map_leaky)
        else:
            act = nn.ReLU()

        if equalized:
            layers = [act, EqualizedLinear(z_dim, w_dim)] * num_layers
        else:
            layers = [act, nn.Linear(z_dim, w_dim)] * num_layers

        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm
        return self.mapping(x)