import pickle
import yaml
import numpy as np
from trainer import Trainer

if __name__ == "__main__":
    ## Training
    # from models import build_model
    # with open('config/exp_config.yaml', 'rb') as f:
    #     config = yaml.load(f)

    # models = build_model(**config["model_params"])
    # data_source_path = config["data_source_path"]
    # data_source_path_valid = None
    # trainer = Trainer(data_source_path, models=models,data_source_path_valid=data_source_path_valid,**config["model_params"])
    # trainer.train()

    # exported_model = trainer.export_log_model()
    # with open(f"artifacts/{exported_model[0][0]}.pkl","wb") as f:
    #     pickle.dump(exported_model[0][1], f)


    ## Predict
    x = np.random.random((3,3200,1))
    model_name = "generator_230615205007"
    with open(f"artifacts/{model_name}.pkl","rb") as f:
        model = pickle.load(f)
    y = model.predict(x)
    print(y)