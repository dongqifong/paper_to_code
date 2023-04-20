from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class Trainer:
    def __init__(self, data_dir,models,**kwargs) -> None:
        self.data_dir = data_dir
        self.preprocessor = models[0]
        self.model = models[1]

        self.test_size = 0.3
        self.shuffle = True
        self.batch_size = 100
        self.epochs = 1

        self.dataloader_train = None
        self.dataloader_valid = None

        self.loss_train = []
        self.loss_valid = []

        self.optim = torch.optim.Adam(self.model.parameters(),lr=1e-4)

        self.load_data()
        pass

    def train(self):
        pass

    def train_one_epoch(self):
        pass

    def valid_one_epoch(self):
        self.model.eval()
        with torch.no_grad():

        pass

    def load_data(self):
        files_list = [i for i in Path(self.data_dir).glob("*.csv")]
        data = np.array([np.transpose(pd.read_csv(f).values) for f in files_list])
        data_train, data_valid = train_test_split(data,test_size=self.test_size,shuffle=self.shuffle)
        dataset_train = ModelDataset(data_train)
        dataset_valid = ModelDataset(data_valid)
        self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size,shuffle=True)
        self.dataloader_valid = DataLoader(dataset_valid, batch_size=self.batch_size,shuffle=False)
    
    def loss_function(self,recons,input,mu,log_var, kld_weight=0.01):
        recons_loss =torch.nn.functional.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

class ModelDataset(Dataset):
    def __init__(self,data, preprocessor) -> None:
        super().__init__()
        self.data = torch.Tensor(data)
        self.preprocessor = preprocessor
        self.data = self.preprocessor(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
        