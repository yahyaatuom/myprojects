# Writing a program, again optimizing the Linear Regression Algorithm in order to predict prices of houses(New Jersey, USA), this time using more than one variable

import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('ML/house_var.csv')
#print(df)

import math
median_bed = math.floor(df.bedrooms.median())
#print(median_bed)


df.bedrooms = df.bedrooms.fillna(median_bed)
#print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

print(reg.coef_)
print(reg.intercept_)

b=reg.predict([[7100,8,3]])
print(b)