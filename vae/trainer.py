import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LossFunc(nn.Module):
    def __init__(self,kld_weight) -> None:
        super().__init__()
        self.kld_weight = kld_weight
        self.loss_recons = 0.0
        self.loss_kld = 0.0

    def forward(self,recons, input, mu, log_var):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.kld_weight * kld_loss

        self.loss_kld = -kld_loss
        self.loss_recons = recons_loss
        return loss




class ModelDataset(Dataset):
    def __init__(self,x) -> None:
        super().__init__()
        self.x = torch.Tensor(x)

    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return len(self.x)
    
def get_dataloader(dataset, batch_size=500, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class Trainer():
    def __init__(self, model, data_dir_train, data_dir_valid, **kwargs) -> None:

        self.data_dir_train = data_dir_train
        self.data_dir_valid = data_dir_valid
        self.model = model

        self.log_metrics = {}
        self.log_parameters = {}
        self.log_models = []

        self.shuffle = kwargs["shuffle"]
        self.epochs = kwargs["epochs"]
        self.lr = kwargs["lr"]
        self.kld_weight = kwargs["kld_weight"]
        self.dataloader_train = self.load_data(self.data_dir_train)
        self.dataloader_valid = self.load_data(self.data_dir_valid)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = LossFunc(self.kld_weight)

        self.loss_train = []
        self.loss_valid = []
        self.loss_train_recons = []
        self.loss_train_kld = []
        self.loss_valid_recons = []
        self.loss_valid_kld = []



    def load_data(self, data_dir):
        from pathlib import Path
        import pandas as pd
        df_list = [pd.read_csv(f) for f in Path(data_dir).glob("*.csv")]
        df = pd.concat(df_list,axis=0)
        x = df.values
        dataset = ModelDataset(x)
        dataloader = get_dataloader(dataset)
        return dataloader

    def train(self):
        for epoch in range(self.epochs):
            self.valid_one_epoch()
            self.train_one_epoch()
            if epoch==0 or (epoch+1)%10==0:
                print(f"[{epoch+1}/{self.epochs}], loss_train=[{self.loss_train[-1]}], loss_valid=[{self.loss_valid[-1]}]")
        self.model.eval()
        loss_recons_train = self.evaluate(self.dataloader_train)
        loss_recons_valid = self.evaluate(self.dataloader_valid)
        self.log_metrics["loss_recons_train"] = loss_recons_train
        self.log_metrics["loss_recons_valid"] = loss_recons_valid
        self.log_models = (["model",self.model,"pytorch"])

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_loss_recons = 0.0
        running_loss_kld = 0.0
        for batch_idx, x in enumerate(self.dataloader_train):
            self.optimizer.zero_grad()
            x_recons = self.model(x)
            loss = self.loss_func(x_recons,x,self.model.mu,self.model.log_var)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            running_loss_recons += self.loss_func.loss_recons.item()
            running_loss_kld += self.loss_func.loss_kld.item()

        self.loss_train.append(round(running_loss/(1+batch_idx),5))
        self.loss_train_recons.append(round(running_loss_recons/(1+batch_idx),5))
        self.loss_train_kld.append(round(running_loss_kld /(1+batch_idx),5))
        return None

    def valid_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        running_loss_recons = 0.0
        running_loss_kld = 0.0
        with torch.no_grad():
            for batch_idx, x in enumerate(self.dataloader_valid):
                x_recons = self.model(x)
                loss = self.loss_func(x_recons,x,self.model.mu,self.model.log_var)
                running_loss += loss.item()
                running_loss_recons += self.loss_func.loss_recons.item()
                running_loss_kld += self.loss_func.loss_kld.item()

            self.loss_valid.append(round(running_loss/(1+batch_idx),5))
            self.loss_valid_recons.append(round(running_loss_recons/(1+batch_idx),5))
            self.loss_valid_kld.append(round(running_loss_kld /(1+batch_idx),5))
        return None
    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss_recons = 0.0
        with torch.no_grad():
            for batch_idx, x in enumerate(dataloader):
                x_recons = self.model(x)
                _ = self.loss_func(x_recons,x,self.model.mu,self.model.log_var)
                running_loss_recons += self.loss_func.loss_recons.item()
        return round(running_loss_recons/(batch_idx+1),5)
        

def test():
    model_params = {}
    data_dir_train = "data_train/"
    data_dir_valid = "data_valid/"

    model_params["shuffle"] = True
    model_params["epochs"] = 10
    model_params["lr"] = 0.001
    model_params["kld_weight"] = 0.1
    from model import VAE
    model = VAE(dim_in=11, dim_latent=2)

    trainer = Trainer(model=model,data_dir_train=data_dir_train,data_dir_valid=data_dir_valid,**model_params)
    trainer.train()
    print(trainer.loss_train_recons)
    print(trainer.loss_train_kld)


if __name__ == "__main__":
    test()