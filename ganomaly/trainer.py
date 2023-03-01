from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models import GLoss, DLoss


class ModelDataset(Dataset):
    def __init__(self, data_source_path, columns_name) -> None:
        super().__init__()
        self.data_source_path = data_source_path
        self.columns_name = columns_name
        self.file_path = [i for i in Path(self.data_source_path).glob("*.csv")]
        self.len_ = len(self.file_path)

    def __getitem__(self, index):
        x = pd.read_csv(self.file_path[index])[self.columns_name].values
        x = torch.Tensor(x).reshape(len(self.columns_name), -1)
        return x

    def __len__(self):
        return self.len_


def get_dataloader(dataset, batch_size=1, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


class Trainer():
    def __init__(self, generator, discriminator, data_source_path, columns_name, batch_size=1, shuffle=False, epochs=1) -> None:
        self.generator = generator
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=1e-4)
        self.discriminator = discriminator
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-6)
        self.data_source = data_source_path
        self.columns_name = columns_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.g_loss_fucn = GLoss()
        self.d_loss_fucn = DLoss()
        self.g_loss_list = []
        self.d_loss_list = []
        self.dataset = ModelDataset(
            self.data_source, columns_name=self.columns_name)
        self.dataloader = get_dataloader(
            self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.real_label = 1
        self.fake_label = 0
        self.epochs = epochs

    def train(self, epochs=None):
        self.generator.train()
        self.discriminator.train()

        if epochs is None:
            epochs = self.epochs

        for epoch in range(epochs):
            running_loss = self.train_d_one_epoch()
            self.d_loss_list.append(running_loss)

            running_loss = self.train_g_one_epoch()
            self.g_loss_list.append(running_loss)

            if (epoch+1) % 10 == 0 or epoch == 0:
                print(
                    f"{epoch+1}/{epochs}, d_loss:{self.d_loss_list[-1]}, g_loss:{self.g_loss_list[-1]}")
        return None

    def train_g_one_epoch(self):
        running_loss = 0.0
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        for batch_idx, x in enumerate(self.dataloader):
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
        for batch_idx, x in enumerate(self.dataloader):
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