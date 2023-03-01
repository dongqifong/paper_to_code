import torch
import numpy as np
from models import anomaly_score


class Predictor():
    def __init__(self, **kwargs) -> None:
        generator_path = kwargs["generator_path"]
        self.generator = torch.load(generator_path)
        self.generator.eval()
        self.z1 = None
        self.z2 = None
        self.x_recon = None
        self.score = 0.0

    def predict(self, x: np.ndarray):
        ## x: (n_samples, n_channels, x_size)
        if len(x.shape) == 3:
            x = torch.Tensor(x)
        else:
            ## check input shape
            return -1

        with torch.no_grad():
            z1, x_recon, z2 = self.generator(x)
            self.score = anomaly_score(z1, z2).numpy()
            self.z1 = z1.numpy()
            self.z2 = z2.numpy()
            self.x_recon = x_recon.numpy()

        return self.score
