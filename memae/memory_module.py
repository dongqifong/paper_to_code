# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:45:58 2022

@author: henry
"""
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def hard_shrink_relu(w, lambd=None, eps=1e-12):
    B, N = w.shape
    if lambd is None:
        lambd = 2 / N
    w_hat = (F.relu(w-lambd) * w) / (torch.abs(w - lambd) + eps)
    return w_hat

class MemModule(nn.Module):
    ''' Memory Module '''
    def __init__(self, mem_dim, feature_dim):
        super().__init__()
        
        self.mem_dim = mem_dim # N
        self.feature_dim = feature_dim # C
        self.memory = Parameter(torch.Tensor(self.mem_dim, self.feature_dim))   # [N, C]
        self.reset_parameters()

    def reset_parameters(self):
        ''' init memory elements : Very Important !! '''
        stdv = 1. / math.sqrt(self.memory.size(1))
        self.memory.data.uniform_(-stdv, stdv)

    def forward(self, z):
        ''' z [B,C] : latent code Z'''
        w = torch.matmul(z, self.memory.T) # [B, C] * [C, N] = [B, N]
        w = F.softmax(w,dim=1)
        
        w_hat = hard_shrink_relu(w) # [B, N]
        # att = F.normalize(w_hat, p=1, dim=1) # [B, N]
        att = F.softmax(w_hat, dim=1) # [B, N]
        z_hat = torch.matmul(att, self.memory) # [B, N] * [N, C] = [B, C]
        return att, z_hat
