import pandas as pd
import numpy as np

for i in range(200):
    x = np.random.random((6400,1))
    df = pd.DataFrame(data=x,columns=["x"])
    df.to_csv(f"data/data_{i}.csv",index=False)