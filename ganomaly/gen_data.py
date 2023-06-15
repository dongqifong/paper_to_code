import pickle
import numpy as np

values = np.random.random((25,3200,4))
datetime = np.random.randint(0,2000, 2000)
with open("data/data_train.pkl","wb") as f:
    pickle.dump({"values":values, "datetime":datetime},f)



values = np.random.random((25,3200,4))
datetime = np.random.randint(0,2000, 2000)
with open("data/data_valid.pkl","wb") as f:
    pickle.dump({"values":values, "datetime":datetime},f)