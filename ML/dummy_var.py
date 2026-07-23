#One Hot Encoding & Dummy Variables

import pandas as pd

df = pd.read_csv("ML/homeprices.csv")
# print(df)

dummies = pd.get_dummies(df.town)
#print(dummies)

merged = pd.concat([df,dummies],axis='columns')
print(merged)