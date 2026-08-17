import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("ML/insurance_data.csv")
#print(df.head)


X_train,X_test,y_train,y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.3)

model = LogisticRegression()
model.fit(X_train,y_train)


score = model.score(X_test,y_test)

print(score)

plt.scatter(df.age,df.bought_insurance,marker='x',color='green')
plt.show()