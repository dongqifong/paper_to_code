import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self,dim_in, dim_latent) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_latent = dim_latent

        self.encoder = nn.Sequential(
            nn.Linear(dim_in,7),
            nn.PReLU(),
            nn.Linear(7,5),
            nn.PReLU(),
            nn.Linear(5,5))
        
        self.fc_mu = nn.Linear(5,dim_latent)
        self.fc_log_var = nn.Linear(5,dim_latent)

        self.decoder = nn.Sequential(
            nn.Linear(dim_latent,7),
            nn.PReLU(),
            nn.Linear(7,5),
            nn.PReLU(),
            nn.Linear(5,dim_in))
        
        self.mu = 0.0
        self.log_var = 0.0
        self.z = 0.0

    def forward(self, x):
        latent = self.encoder(x)
        self.mu = self.fc_mu(latent)
        self.log_var = self.fc_log_var(latent)
        self.z = self.reparameterize(self.mu, self.log_var)
        x_recons = self.decoder(self.z)
        return x_recons

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
def test(dim_in, dim_latent):
    import torch
    dim_in = 10
    dim_latent = 3
    model = VAE(dim_in, dim_latent)

    x = torch.randn((100,10))
    x_recons = model(x)
    print(x_recons.shape)
    return None

if __name__ == "__main__":
    import torch
    dim_in = 10
    dim_latent = 2
    test(dim_in, dim_latent)