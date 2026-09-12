import pandas as pd

df = pd.read_csv("ML/test_scores.csv")
Nan = df.isnull().sum()
print(Nan)
