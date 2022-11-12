# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:29:34 2022

@author: henry
"""

import torch
import torch.nn.functional as F

def mem_loss(x,x_recon,att,alpha=0.0002, eps=1e-12):
    r = F.mse_loss(x_recon, x) # mse loss

    e = att * torch.log(att + eps) # entropy loss
    e = -1.0 * torch.sum(e, dim=1)
    e = e.mean()
    return r + alpha*e