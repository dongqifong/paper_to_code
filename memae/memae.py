# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:33:24 2022

@author: henry
"""

import torch.nn as nn
from memory_module import MemModule


class Encoder(nn.Module):
    def __init__(self, x_size, feature_dim, bias=False):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(x_size, x_size//2, bias=bias),
            nn.PReLU(),
            nn.Linear(x_size//2, x_size//2, bias=bias),
            nn.PReLU(),
            nn.Linear(x_size//2, feature_dim, bias=bias)
            )
    def forward(self,x):
        return self.linear_block(x)

class Decoder(nn.Module):
    def __init__(self, x_size, feature_dim, bias=False):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(feature_dim, x_size//2, bias=bias),
            nn.PReLU(),
            nn.Linear(x_size//2, x_size//2, bias=bias),
            nn.PReLU(),
            nn.Linear(x_size//2, x_size, bias=bias)
            )
    def forward(self,x):
        return self.linear_block(x)

class MemAE(nn.Module):
    def __init__(self, x_size, mem_dim, feature_dim):
        super().__init__()
        self.encoder = Encoder(x_size, feature_dim)
        self.memory_block = MemModule(mem_dim, feature_dim)
        self.decoder = Decoder(x_size, feature_dim)
        
    def forward(self,x):
        z = self.encoder(x)
        att, z_hat = self.memory_block(z)
        x_recon = self.decoder(z_hat)
        return att, x_recon
