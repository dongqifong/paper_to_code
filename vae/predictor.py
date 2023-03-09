import torch

class Predictor():
    def __init__(self,**kwargs) -> None:
        self.model = kwargs["model"]
        self.model.eval()
        pass

    def predict(self,x):
        with torch.no_grad():
            x_recons = self.model(torch.Tensor(x))
            diff = torch.mean((x_recons-x)**2,dim=1)
        return diff.numpy()
    
def test(dim_in, dim_latent):
    import torch
    import numpy as np
    from model import VAE
    dim_in = 10
    dim_latent = 3
    model = VAE(dim_in, dim_latent)
    model_dict = {}
    model_dict["model"] = model
    predictor = Predictor(**model_dict)

    x = np.random.random((100,10))
    print(predictor.predict(x))
    return None

if __name__ == "__main__":
    import torch
    dim_in = 10
    dim_latent = 2
    test(dim_in, dim_latent)