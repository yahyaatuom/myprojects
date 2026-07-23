import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("ML/homeprices.csv")

dummies = pd.get_dummies(df.town)
merged = pd.concat([df, dummies], axis='columns')
final = merged.drop(['town', 'west windsor'], axis='columns')

model = LinearRegression()
x = final.drop('price', axis='columns')
y = final.price
model.fit(x, y)

le = LabelEncoder()
dfle = df.copy()
dfle.town = le.fit_transform(dfle.town)

X = dfle[['town', 'area']].values
Y = dfle[['town', 'area', 'price']].values

ohe = OneHotEncoder(sparse_output=False)
X = ohe.fit_transform(X)
#print(X)

X = X[:,1:]
print(X)

model.fit(X,y)