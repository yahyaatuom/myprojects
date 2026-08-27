import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

hours_slept = [4,5,5.5,6,6.5,7,7.5,8,8.5,9]
energy_scores = [32,38,41,45,47,49,58,67,78,90]

X = [[hours] for hours in hours_slept]
y = energy_scores

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = SVC()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test,predictions)
print(f"Predicted scores:  {predictions}")
print(f"Actual Score: ", y_test)
print(f"Mean Squared Error: {mse}")

score = model.score(X_test,y_test)
print(f"Model prediction accuracy: {score}")

plt.scatter(X,y,color="blue",label= "Real data")
plt.plot(X, model.predict(X),color = "green", label= "Model Prediction")
plt.xlabel("Hours Slept")
plt.ylabel("Energy scores")
plt.title("Hours Slept vs Energy Scores")
plt.legend()
plt.show()