import numpy as np
import pandas as pd

x = np.random.random(10240)
data = {"a":x}
df = pd.DataFrame(data=data)

for i in range(20):
    df.to_csv(f"data/data_{i}.csv",index=False,encoding="utf_8_sig")