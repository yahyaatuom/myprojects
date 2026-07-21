# This Program will train a model using the records in csv optimizing the Linear Regression Algorithm to Predict the Per Capita Income of Canada in The Year 2020
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Load dataset
df = pd.read_csv("ML/canada_per_capita_income.csv")

reg = linear_model.LinearRegression()
reg.fit(df[["year"]], df["per capita income (US$)"])

predicted_income = reg.predict([[2020]])
print(predicted_income)