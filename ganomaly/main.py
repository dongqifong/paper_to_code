from models import get_gen_dis
from trainer import Trainer
import torch

if __name__ == "__main__":
    data_source_path = 'data/'
    in_channels = 1
    x_size = 10240
    kernal_size = 32
    stride = 8

    ## Training
    generator, discriminator = get_gen_dis(
        in_channels, x_size, kernal_size, stride)
    columns_name = "a"
    batch_size = 10
    shuffle = False
    trainer = Trainer(generator=generator, discriminator=discriminator, data_source_path=data_source_path,
                      columns_name=columns_name, batch_size=batch_size, shuffle=shuffle)
    trainer.train(1)
    # save model
    torch.save(trainer.generator, "saved_models/generator.pth")
    torch.save(trainer.discriminator, "saved_models/discriminator.pth")

    ## Prediction
    from predictor import Predictor
    import numpy as np
    kwargs = {"generator_path": "saved_models/generator.pth"}
    predictor = Predictor(**kwargs)
    x = np.random.random((10, in_channels, x_size))
    score = predictor.predict(x)
    print(score)
