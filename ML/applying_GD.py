import numpy as np
import pandas as pd

df = pd.read_csv("ML/test_scores.csv")
print(df)

x = df['math'].values
y=df['cs'].values

#feature scaling, scaling the data so learning rate doesn't cause divergence
x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)

x_scaled = (x-x_mean) / x_std
y_scaled = (y-y_mean) /y_std

m_curr = b_curr = 0
iteration = 10000
learning_rate = 0.01
n = len(x_scaled)

for i in range(iteration):
    y_predicted = m_curr * x_scaled +b_curr
    md = -(2/n) * np.sum(x_scaled * (y_scaled - y_predicted))
    bd = -(2/n) *np.sum(y_scaled - y_predicted)

    m_curr -= learning_rate *md
    b_curr -= learning_rate *bd

    print(m_curr)
