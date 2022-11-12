# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:30:59 2022

@author: henry
"""

import torch
import trainer
import memae

x_size = 10
mem_dim = 100
feature_dim = 5
alpha = 0.1
x = torch.randn((128,x_size))

model = memae.MemAE(x_size,mem_dim,feature_dim)
model_trainer = trainer.Trainer(x_train=x,model=model,alpha=alpha)
model_trainer.train(10)
model_trainer.training_loss
