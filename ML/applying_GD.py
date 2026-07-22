import numpy as np
import pandas as pd

df = pd.read_csv("ML/test_scores.csv")
print(df)

x = df['maths'].values
y=df['cs'].values

