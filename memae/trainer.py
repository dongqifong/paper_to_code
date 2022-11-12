# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:18:19 2022

@author: henry
"""
from model_loss_func import mem_loss
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Trainer:
    def __init__(self, x_train, y_train=None, x_valid=None, y_valid=None, model=None, lr=1e-4,shuffle=False, batch_size=128, alpha=0.02):
        self.model = model
        self.lr =lr
        
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.training_loss = [-1]
        self.valid_loss = [-1]
        
        self.show_progess = 10
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        self.dataset_train = ModelDataset(x_train, y_train)
        self.dataloader_train = DataLoader(self.dataset_train, shuffle=self.shuffle, batch_size=self.batch_size)
        
        if x_valid:
            self.dataset_valid = ModelDataset(x_valid, y_valid)
            self.dataloader_valid = DataLoader(self.dataset_valid, shuffle=self.shuffle, batch_size=self.batch_size)
        else:
            self.dataset_valid = None
            self.dataloader_valid = None
        
        self.loss_func = mem_loss
        self.alpha = alpha
    
    def show_training_valid_loss(self,epoch,epochs):
        if epoch==0 or (epoch+1 % self.show_progess)==0:
            print(f"Progress:[{epoch+1}/{epochs}], Training loss:{self.training_loss[-1]}, Validation loss:[{self.valid_loss[-1]}]")
        return None
    
    def train_one_epoch(self):
        self.model.train()
        self.model.to(self.device)
        running_loss = 0.0
        for batch_idx, (x,y,_) in enumerate(self.dataloader_train):
            self.optim.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            att, x_recon = self.model(x)
            loss = self.loss_func(x,x_recon,att,alpha=self.alpha)
            loss.backward()
            self.optim.step()
            running_loss += loss.cpu().item()
        running_loss /= (batch_idx+1)
        return running_loss
    
    def valid_one_epoch(self):
        self.model.eval()
        self.model.to(self.device)
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x,y,_) in enumerate(self.dataloader_valid):
                x = x.to(self.device)
                y = y.to(self.device)
                att, x_recon = self.model(x)
                loss = self.loss_func(x,x_recon,att,alpha=self.alpha)
                running_loss += loss.cpu().item()
            running_loss /= (batch_idx+1)
        return running_loss
    
    def train(self, epochs):
        for epoch in range(epochs):
            if self.dataloader_valid is not None:
                valid_loss = self.valid_one_epoch()
                self.valid_loss.append(round(valid_loss,5))
                
            if self.dataloader_train is not None:
                trainining_loss = self.train_one_epoch()
                self.training_loss.append(round(trainining_loss,5))
                
            self.show_training_valid_loss(epoch,epochs)
        return None
    
class ModelDataset(Dataset):
    def __init__(self,x ,y=None):
        self.x = torch.Tensor(x)
        
        if y is not None:
            self.y = torch.Tensor(y)
        else:
            self.y = torch.ones(len(self.x))
                    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx], idx
        
    def __len__(self):
        return len(self.x)
            
    

