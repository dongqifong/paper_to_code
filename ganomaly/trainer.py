import numpy as np
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models import GLoss, DLoss
import pickle
import datetime

class ModelDataset(Dataset):
    def __init__(self, data_source_path, columns_idx) -> None:
        super().__init__()

        self.data_source_path = data_source_path
        self.columns_index = columns_idx

        with open(f"data/{self.data_source_path}.pkl","rb") as f:
                data = pickle.load(f) # key: datetime, values

        self.len_ = len(data["datetime"])

        self.x = torch.Tensor(data["values"])[:,:,self.columns_index]
        self.x = self.x.unsqueeze(1)
        print(self.x.shape)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len_


def get_dataloader(dataset, batch_size=1, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


class Trainer():
    def __init__(self, data_source_path, models, data_source_path_valid=None,**kwargs) -> None:
        self.generator = models[0]
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=kwargs["lr_g"])
        
        self.discriminator = models[1]
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=kwargs["lr_d"])
        
        self.data_source_path = data_source_path
        self.columns_idx = kwargs["columns_idx"] # user input
        self.batch_size = kwargs["batch_size"] # user input
        self.shuffle = kwargs["shuffle"] # user input
        self.g_loss_fucn = GLoss()
        self.d_loss_fucn = DLoss()
        self.g_loss_list = []
        self.d_loss_list = []

        self.dataset_train = ModelDataset(
            self.data_source_path, columns_idx=self.columns_idx)
        self.dataloader_train = get_dataloader(
            self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)
        
        if data_source_path_valid is not None:
            self.dataset_valid = ModelDataset(
                data_source_path_valid, columns_idx=self.columns_idx)
            self.dataloader_valid = get_dataloader(
                self.dataset_valid, batch_size=self.batch_size, shuffle=self.shuffle)
        else:
            self.dataset_valid = None
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.real_label = 1
        self.fake_label = 0
        self.epochs = kwargs["epochs"] # user input

    def train(self, epochs=None):
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.epochs):
            running_loss = self.train_d_one_epoch()
            self.d_loss_list.append(running_loss)

            running_loss = self.train_g_one_epoch()
            self.g_loss_list.append(running_loss)

            if (epoch+1) % 10 == 0 or epoch == 0:
                print(
                    f"{epoch+1}/{self.epochs}, d_loss:{self.d_loss_list[-1]}, g_loss:{self.g_loss_list[-1]}")
        return None

    def train_g_one_epoch(self):
        running_loss = 0.0
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        for batch_idx, x in enumerate(self.dataloader_train):
            x = x.to(self.device)
            self.optimizer_g.zero_grad()
            z1, x_recon, z2 = self.generator(x)
            y_mid_z1, y_z1 = self.discriminator(z1)
            y_mid_z2, y_z2 = self.discriminator(z2)
            loss = self.g_loss_fucn(x, x_recon, z1, z2, y_mid_z1, y_mid_z2)
            loss.backward()
            self.optimizer_g.step()
            running_loss += loss.detach().cpu().item()

        return round(running_loss/(batch_idx+1), 5)

    def train_d_one_epoch(self):
        running_loss = 0.0
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        for batch_idx, x in enumerate(self.dataloader_train):
            x = x.to(self.device)
            self.optimizer_d.zero_grad()
            z1, _, z2 = self.generator(x)
            _, y_z1 = self.discriminator(z1.detach())
            _, y_z2 = self.discriminator(z2.detach())

            label_real = torch.full(
                (z1.shape[0],), self.real_label, dtype=torch.long, device=self.device)
            label_fake = torch.full(
                (z2.shape[0],), self.fake_label, dtype=torch.long, device=self.device)

            loss = self.d_loss_fucn(torch.concat(
                [y_z1, y_z2]), torch.concat([label_real, label_fake]))
            loss.backward()
            self.optimizer_d.step()
            running_loss += loss.detach().cpu().item()

        return round(running_loss/(batch_idx+1), 5)
    
    def export_log_model(self):
        self.generator.cpu().eval()
        self.discriminator.cpu().eval()
        from predictor import Predictor
        predictor_param = {"generator": self.generator}
        predictor = Predictor(**predictor_param)

        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        torch.save(self.generator, f"artifacts/generator_{timestamp}.pth")
        return [(f"generator_{timestamp}",predictor,"pyfunc")]
    
# if __name__=="__main__":
#     import yaml

#     from models import build_model
#     with open('config/exp_config.yaml', 'rb') as f:
#         config = yaml.load(f)

#     models = build_model(**config["model_params"])
#     data_source_path = config["data_source_path"]
#     data_source_path_valid = None
#     trainer = Trainer(data_source_path, models=models,data_source_path_valid=data_source_path_valid,**config["model_params"])
#     trainer.train()

#     exported_model = trainer.export_log_model()
#     with open("artifacts/generator.pkl","wb") as f:
#         pickle.dump(exported_model[0][1], f)