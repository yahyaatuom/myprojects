# Writing a program, again optimizing the Linear Regression Algorithm in order to predict prices of houses(New Jersey, USA), this time using more than one variable

import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('ML/house_var.csv')
print(df)