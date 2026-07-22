import numpy as np
import pandas as pd

df = pd.read_csv("ML/test_scores.csv")
print(df)

x = df['maths'].values
y=df['cs'].values

#feature scaling, scaling the data so learning rate doesn't cause divergence
x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)