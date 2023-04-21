from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import build_model

class Trainer:
    def __init__(self, data_source, models,**kwargs) -> None:
        self.data_source = data_source
        self.preprocessor = models[0]
        self.model = models[1]

        self.test_size = kwargs["test_size"]
        self.shuffle = kwargs["shuffle"]
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]

        self.dataloader_train = None
        self.dataloader_valid = None

        self.loss_train = []
        self.loss_train_recons = []
        self.loss_train_kld = []

        self.loss_valid = []
        self.loss_valid_recons = []
        self.loss_valid_kld = []

        self.optim = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        self.load_data()

    def train(self):
        for epoch in range(self.epochs):
            running_loss, loss_recons, loss_kld = self.valid_one_epoch()
            self.loss_valid.append(running_loss)
            self.loss_valid_recons.append(loss_recons)
            self.loss_valid_kld.append(loss_kld)

            running_loss, loss_recons, loss_kld = self.train_one_epoch()
            self.loss_train.append(running_loss)
            self.loss_train_recons.append(loss_recons)
            self.loss_train_kld.append(loss_kld)

            if epoch==0 or (epoch+1)%10==0:
                print(f"[{epoch+1}]/[{self.epochs}], (loss_train,loss_train_recons,loss_train_kld)=({self.loss_train[-1]},{self.loss_train_recons[-1]},{self.loss_train_kld[-1]}),(loss_valid,loss_valid_recons,loss_valid_kld)=({self.loss_valid[-1]},{self.loss_valid_recons[-1]},{self.loss_valid_kld[-1]})")

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        loss_recons = 0.0
        loss_kld = 0.0
        for batch_idx, x in enumerate(self.dataloader_train):
            self.optim.zero_grad()
            z, x_recons, mu, log_var = self.model(x)
            loss = self.loss_function(x_recons,x,mu,log_var)
            loss["loss"].backward()
            self.optim.step()
            running_loss += loss["loss"].item()
            loss_recons += loss["loss_recons"].item()
            loss_kld += loss["kld"].item()
        running_loss = round(running_loss/(batch_idx+1),5)
        loss_recons = round(loss_recons/(batch_idx+1),5)
        loss_kld = round(loss_kld /(batch_idx+1),5)
        return running_loss, loss_recons, loss_kld

    def valid_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        loss_recons = 0.0
        loss_kld = 0.0
        with torch.no_grad():
            for batch_idx, x in enumerate(self.dataloader_valid):
                z, x_recons, mu, log_var = self.model(x)
                loss = self.loss_function(x_recons,x,mu,log_var)
                running_loss += loss["loss"].item()
                loss_recons += loss["loss_recons"].item()
                loss_kld += loss["kld"].item()
            running_loss = round(running_loss/(batch_idx+1),5)
            loss_recons = round(loss_recons/(batch_idx+1),5)
            loss_kld = round(loss_kld /(batch_idx+1),5)
        return running_loss, loss_recons, loss_kld

    def load_data(self):
        files_list = [i for i in Path(self.data_source).glob("*.csv")]
        data = np.concatenate([np.transpose(pd.read_csv(f).values) for f in files_list])
        data_train, data_valid = train_test_split(data,test_size=self.test_size,shuffle=self.shuffle)
        dataset_train = ModelDataset(data_train,self.preprocessor)
        dataset_valid = ModelDataset(data_valid,self.preprocessor)
        self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size,shuffle=True)
        self.dataloader_valid = DataLoader(dataset_valid, batch_size=self.batch_size,shuffle=False)
    
    def loss_function(self,recons,input,mu,log_var, kld_weight=0.01):
        recons_loss =torch.nn.functional.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = (1,2)), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'loss_recons':recons_loss.detach(), 'kld':-kld_loss.detach()}

class ModelDataset(Dataset):
    def __init__(self,data, preprocessor) -> None:
        super().__init__()
        self.data = torch.Tensor(data).unsqueeze(dim=1)
        self.preprocessor = preprocessor
        self.data = self.preprocessor(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
        
x_size = 6400
in_channels = 1
models = build_model(**{"x_size":x_size,"in_channels":in_channels})
data_source = "data/"

model_parameters = {}
model_parameters["test_size"] = 0.3
model_parameters["shuffle"] = True
model_parameters["batch_size"] = 100
model_parameters["epochs"] = 1

trainer = Trainer(data_source, models, **model_parameters)
trainer.train()