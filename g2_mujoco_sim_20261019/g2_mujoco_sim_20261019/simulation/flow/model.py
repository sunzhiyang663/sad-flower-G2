"""Small native PyTorch Transformer baseline for G2 CFM, not SAD-Flower weights."""
import math

import torch
from torch import nn


class G2Transformer(nn.Module):
    def __init__(self, horizon=64, width=128, depth=4, heads=4):
        super().__init__()
        if width % heads or width % 2:
            raise ValueError('Width must be even and divisible by attention heads')
        self.horizon = horizon
        self.input = nn.Linear(6, width)
        self.position = nn.Parameter(torch.randn(1, horizon, width)*.02)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.register_buffer('frequencies', torch.exp(
            -math.log(10000)*torch.arange(width//2)/(width//2)))
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width*4,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 6))

    def forward(self, x, tau):
        phase = tau.reshape(-1, 1)*self.frequencies[None]*1000
        embedding = self.time(torch.cat([phase.sin(), phase.cos()], dim=-1))
        return self.output(self.encoder(self.input(x)+self.position+embedding[:, None]))
