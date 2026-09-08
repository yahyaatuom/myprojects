import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

digits = load_digits()
df = pd.DataFrame(digits.data)

df['target'] = digits.target

X_train,X_test,y_train,y_test = train_test_split(
    df.drop(['target'],axis='columns'),
    digits.target,
    test_size=0.2
)

model = RandomForestClassifier()
model.fit(X_train,y_train)
a = model.score(X_test,y_test)
print(a)

#-- Confusion Matrix