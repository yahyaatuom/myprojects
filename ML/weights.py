import numpy as np
from sklearn.linear_model import LogisticRegression

#(Age, Income, Credit Score)
X = np.array([[25, 40000, 600], [45, 80000, 750], [22, 18000, 520], [50, 120000, 780]])
y = np.array([0, 1, 0, 1]) # Class 0 or Class 1


model = LogisticRegression()
model.fit(X, y) 

# 3. Pull out the automatically calculated numbers
print("Automatically calculated Weights (W):", model.coef_)
print("Automatically calculated Bias (b):", model.intercept_)