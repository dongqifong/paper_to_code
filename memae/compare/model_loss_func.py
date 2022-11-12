# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:29:34 2022

@author: henry
"""

import torch
import torch.nn.functional as F

def mem_loss(x,x_recon,att=0,alpha=0.0002, eps=1e-12):
    r = F.mse_loss(x_recon, x) # mse loss
    return r