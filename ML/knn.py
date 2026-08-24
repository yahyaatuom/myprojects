from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
#print(df.head)


X = df.drop(['target'],axis='columns')
y = df.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train,y_train)
score = knn.score(X_test,y_test)
print(score)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

#--- AUC---
y_prob = knn.predict_proba(X_test)

auc = roc_auc_score(y_test,y_prob,multi_class="ovr")
print(f"AUC: {auc}")