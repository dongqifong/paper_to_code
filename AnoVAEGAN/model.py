import torch
import torch.nn as nn

def build_model(**kwargs):
    x_size = kwargs["x_size"]
    in_channels = kwargs["in_channels"]
    preprocessor = Preprocessor(in_channels)
    model = SVAE(x_size,in_channels)
    return (preprocessor, model)

class Preprocessor(nn.Module):
    def __init__(self,num_features) -> None:
        super().__init__()
        self.func = nn.InstanceNorm1d(num_features=num_features)
    def forward(self,x):
        return self.func(x)

class Encoder(nn.Module):
    def __init__(self,x_size,in_channels) -> None:
        super().__init__()
        self.x_size = x_size
        self.in_channels = in_channels
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels,10*in_channels,kernel_size=64,stride=8),
            nn.PReLU(),
            nn.InstanceNorm1d(10*in_channels),
            nn.Conv1d(10*in_channels,20*in_channels,kernel_size=32,stride=4),
            nn.PReLU(),
            nn.InstanceNorm1d(20*in_channels),
            nn.Conv1d(20*in_channels,40*in_channels,kernel_size=16,stride=4),
            nn.PReLU(),
            nn.InstanceNorm1d(40*in_channels),
            nn.Conv1d(40*in_channels,80*in_channels,kernel_size=4,stride=2))
    def forward(self,x):
        x = self.cnn1(x)
        return x

class Decoder(nn.Module):
    def __init__(self,x_size,in_channels) -> None:
        super().__init__()
        self.x_size = x_size
        self.in_channels = in_channels
        self.cnn1 = nn.Sequential(
            nn.ConvTranspose1d(40*in_channels,20*in_channels,kernel_size=4,stride=2),
            nn.PReLU(),
            nn.InstanceNorm1d(20*in_channels),
            nn.ConvTranspose1d(20*in_channels,10*in_channels,kernel_size=16,stride=4),
            nn.PReLU(),
            nn.InstanceNorm1d(10*in_channels),
            nn.ConvTranspose1d(10*in_channels,5*in_channels,kernel_size=32,stride=4),
            nn.PReLU(),
            nn.InstanceNorm1d(5*in_channels),
            nn.ConvTranspose1d(5*in_channels,in_channels,kernel_size=64,stride=8),
            nn.AdaptiveAvgPool1d(self.x_size))
    
    def forward(self,z):
        x_recons = self.cnn1(z)
        return x_recons

class SVAE(nn.Module):
    def __init__(self,x_size,n_channels) -> None:
        super().__init__()
        self.encoder = Encoder(x_size,n_channels)
        self.decoder = Decoder(x_size,n_channels)
        pass

    def forward(self,x):
        x = self.encoder(x)
        mu = x[:,:(x.shape[1]//2)]
        log_var = x[:,(x.shape[1]//2):]
        z = self.reparameterize(mu, log_var)
        x_recons = self.decoder(z)
        return z, x_recons, mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

# x_size = 6400
# n_channels = 1
# x = torch.randn((1,n_channels,x_size))
# encoder = Encoder(x_size,n_channels)
# decoder = Decoder(x_size,n_channels)
# z = encoder(x)
# x_recons = decoder(z[:,:40])
# print(x_size)
# print(z.shape)
# print(x_recons.shape)

# model = SVAE(x_size,n_channels)
# z, x_recons = model(x)
# print(z.shape)
# print(x_recons.shape)
