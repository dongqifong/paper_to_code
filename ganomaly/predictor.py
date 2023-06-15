import torch
import numpy as np
# from models import anomaly_score


class Predictor():
    def __init__(self, **kwargs) -> None:
        # generator_path = kwargs["generator_path"]
        # self.generator = torch.load(generator_path)
        self.generator = kwargs["generator"]
        self.generator.eval()
        self.z1 = None
        self.z2 = None
        self.x_recon = None
        self.score = 0.0

    @torch.no_grad()
    def predict(self, x: np.ndarray):
        ## x: (n_samples, x_size, n_channels)
        if len(x.shape) == 3:
            x = torch.Tensor(x).permute((0,2,1))

        elif len(x.shape) == 2:
            x = torch.Tensor(x).unsqueeze(0).permute((0,2,1))
        else:
            ## check input shape
            return -1

        z1, x_recon, z2 = self.generator(x)
        self.score = self.anomaly_score(z1, z2).numpy()
        self.z1 = z1.numpy()
        self.z2 = z2.numpy()
        self.x_recon = x_recon.numpy()
        return self.score

    def anomaly_score(self, z_1, z_2):
        score = torch.mean(torch.abs(z_1 - z_2), dim=(1, 2))
        return score
    

# if __name__ == "__main__":
#     import pickle
#     x = np.random.random((3,3200,1))
#     with open("artifacts/generator.pkl","rb") as f:
#         model = pickle.load(f)
    
#     y = model.predict(x)
#     print(y)