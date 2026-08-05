from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Linear regression with cross-validation
lr = LinearRegression()
scores = cross_val_score(lr, X, y, cv=5)

print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.3f}")
print(f"Feature names: {diabetes.feature_names}")