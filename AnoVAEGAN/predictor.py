import torch
import numpy as np

class Predictor:
    def __init__(self,**kwargs) -> None:
        self.preprocessor = kwargs["preprocessor"]
        self.model = kwargs["model"]

    def predict(self, x):
        x = torch.Tensor(x).reshape(1,1,x.shape[-1])
        z, x_recons = self.model(x)
        anomaly_score = torch.mean(torch.abs(x-x_recons), dim=(1,2))
        return anomaly_score.numpy()