# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:38:34 2022

@author: henry
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
        

class Encoder(nn.Module):
    def __init__(self, in_channels, x_size, kernel_size=64, stride=8):
        super().__init__()
        self.in_channels = in_channels
        self.x_size = x_size
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels*32, kernel_size=self.kernel_size, stride=self.stride),
            nn.PReLU(),
            nn.Dropout1d(p=0.2),
            nn.Conv1d(in_channels=self.in_channels*32, out_channels=self.in_channels*64, kernel_size=self.kernel_size, stride=self.stride),
            nn.PReLU(),
            nn.Dropout1d(p=0.2),
            nn.Conv1d(in_channels=self.in_channels*64, out_channels=self.in_channels*128, kernel_size=self.kernel_size, stride=self.stride),
            nn.PReLU()
        )

    def forward(self,x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, x_size, kernel_size=64, stride=8):
        super().__init__()
        self.in_channels = in_channels
        self.x_size = x_size
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.in_channels*128, out_channels=self.in_channels*64, kernel_size=self.kernel_size, stride=self.stride),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.in_channels*64, out_channels=self.in_channels*32, kernel_size=self.kernel_size, stride=self.stride),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.in_channels*32, out_channels=self.in_channels*16, kernel_size=self.kernel_size, stride=self.stride),
            nn.AdaptiveAvgPool1d(self.x_size),
            nn.Conv1d(in_channels=self.in_channels*16, out_channels=self.in_channels,kernel_size=1)
        ) 

    def forward(self,x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self,in_channels,x_size,kernel_size,stride):
        super().__init__()
        self.encoder1 = Encoder(in_channels,x_size,kernel_size=kernel_size,stride=stride)
        self.decoder1 = Decoder(in_channels,x_size,kernel_size=kernel_size,stride=stride)
        self.encoder2 = Encoder(in_channels,x_size,kernel_size=kernel_size,stride=stride)

    def forward(self,x):
        z1 = self.encoder1(x)
        x_recon = self.decoder1(z1)
        z2 = self.encoder2(x_recon)
        return z1, x_recon, z2

class Discriminator(nn.Module):
    def __init__(self, z_shape):
        super().__init__()
        flatten_size = z_shape[1]*z_shape[2]

        self.fc1 = nn.Sequential(
            nn.Linear(flatten_size, flatten_size//2),
            nn.PReLU())

        self.fc2 = nn.Linear(flatten_size//2, 2, bias=False)

    def forward(self,z):
        y_mid = self.fc1(torch.flatten(z,start_dim=1))
        y = self.fc2(y_mid)
        return y_mid, y

def get_gen_dis(in_channels,x_size,kernel_size,stride):
    generator = Generator(in_channels,x_size,kernel_size=kernel_size,stride=stride)
    z1, _, _ = generator(x)
    discriminator = Discriminator(z1.shape)
    return generator, discriminator

class EncLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = nn.MSELoss()
    def forward(self,z1,z2):
        return self.func(z1,z2)

class ConLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = nn.L1Loss()
    def forward(self,x,x_recon):
        return self.func(x,x_recon)

class AdvLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = nn.MSELoss()

    def forward(self,y_mid_z1,y_mid_z2):
        return self.func(y_mid_z1,y_mid_z2)

class GLoss(nn.Module):
    def __init__(self,w_enloss=1,w_conloss=1,w_advloss=1) -> None:
        super().__init__()
        self.w_enloss = w_enloss
        self.w_conloss = w_conloss
        self.w_advloss = w_advloss
        self.enloss_func = EncLoss()
        self.conloss_func = ConLoss()
        self.advloss_func = AdvLoss()

    def forward(self,x,x_recon,z1,z2,y_mid_z1,y_mid_z2):
        self.enloss = self.enloss_func(z1,z2)
        self.conloss = self.conloss_func(x,x_recon)
        self.advloss = self.advloss_func(y_mid_z1,y_mid_z2)
        return self.w_enloss*self.enloss + self.w_conloss*self.conloss + self.w_advloss*self.advloss

class DLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = nn.CrossEntropyLoss()
        return None

    def forward(self,y_pred, y_true):
        return self.func(y_pred, y_true)

if __name__ == "__main__":
    in_channels = 1
    x_size = 6400
    kernel_size = 64
    stride = 8

    x = torch.randn((2,in_channels,x_size))

    generator, discriminator = get_gen_dis(in_channels,x_size,kernel_size,stride)
    z1, x_recon, z2 = generator(x)
    y_mid_z1, y_z1 = discriminator(z1)
    y_mid_z2, y_z2 = discriminator(z2)

    enloss_func = EncLoss()
    conloss_func = ConLoss()
    advloss_func = AdvLoss()

    enloss = enloss_func(z1,z2)
    conloss = conloss_func(x,x_recon)
    advloss = advloss_func(y_mid_z1,y_mid_z2)

    print(enloss)
    print(conloss)
    print(advloss)

    gloss = GLoss()
    running_loss = gloss(x,x_recon,z1,z2,y_mid_z1,y_mid_z2)
    print(running_loss)