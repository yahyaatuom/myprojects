from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
#print(df.head)


X = df.drop(['target'],axis='columns')
y = df.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train,y_train)
score = knn.score(X_test,y_test)
print(score)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)