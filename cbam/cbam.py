#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 22:14:45 2022

@author: henry
"""

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=None):
        super(ChannelAttention, self).__init__()
        
        if ratio is None:
            ratio = 2
            
        if ratio>in_channels:
            ratio = in_channels
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv1d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size = kernel_size -1
        padding = kernel_size // 2

        self.conv = nn.Conv1d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
    
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=None, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention(kernel_size)
        self.Mc = None
        self.Ms = None
        
    def forward(self,x):
        self.Mc = self.ca(x)
        x = self.Mc * x
        self.Ms = self.sa(x)
        x = self.Ms * x
        return x
    
if __name__ == "__main__":
    x = torch.randn((10,32,1000))
    m = CBAM(in_channels=32,ratio=1.1,kernel_size=3)
    mask = m(x)
    print(mask.shape)
    y = mask * x
    print(y.shape)
