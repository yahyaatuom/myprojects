#Simple Linear Regression model that'd predict home prices in BatKhela, KPK, Pakistan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model  # The most important library here

df = pd.read_csv("ML/houses.csv", sep="\t")
#print(df)

# plt.scatter(df.area, df.price, color = "red", marker = "+")
# plt.xlabel("Area (Sq.Ft)")
# plt.ylabel("Price(PKR)")

# #Training the Model

reg = linear_model.LinearRegression()
reg.fit(df[["area"]], df.price)

# print(reg.predict([[50000]]))

#print(reg.coef_)
#print(reg.intercept_)

#print(7481.08108108*5000+-6867837.837837841)
plt.xlabel('area', fontsize=20)
plt.ylabel('price',fontsize=20)
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color = 'blue')
#plt.show()
# d = pd.read_csv("area.csv")
# d.head(3)

# p = reg.predict(d)
# d['prices'] = p
# d.to_csv('prediction.csv',index=False)

import pickle
with open('model_pickle','wb') as f:
    pickle.dump(reg, f)

print("Success! Model trained and saved as model_pickle.")