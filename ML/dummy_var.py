#One Hot Encoding & Dummy Variables

import pandas as pd

df = pd.read_csv("ML/homeprices.csv")
# print(df)

dummies = pd.get_dummies(df.town)
#print(dummies)

merged = pd.concat([df,dummies],axis='columns')
#print(merged)

final = merged.drop(['town','west windsor'], axis='columns')
#print(final)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

x = final.drop('price',axis='columns')
y = final.price

model.fit(x,y)