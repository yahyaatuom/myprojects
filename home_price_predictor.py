#Simple Linear Regression model that'd predict home prices in BatKhela, KPK, Pakistan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model  # The most important library here

df = pd.read_csv("houses.csv", sep="\t")
print(df)

plt.scatter(df.area, df.price)
plt.show()