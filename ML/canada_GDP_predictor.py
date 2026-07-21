#This Program will train a model using the records in csv optimizing the Linear Regression Algorithm to Predict the Per Capita Income of Canada in The Year 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("ML/canada_per_capita_income.csv")

print(df)