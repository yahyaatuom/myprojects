import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import joblib
import os

print("Working directory:", os.getcwd())

df = pd.read_csv("ML/canada_per_capita_income.csv")

reg = linear_model.LinearRegression()
reg.fit(df[["year"]], df[["per capita income (US$)"]])

# Predict
print(reg.predict(pd.DataFrame([[2026]], columns=["year"])))

# Save model
joblib.dump(reg, "ML/model_joblib.pkl")

# Load model
b = joblib.load("ML/model_joblib.pkl")

# Predict again
print(b.predict(pd.DataFrame([[2027]], columns=["year"])))


print(b.coef_)