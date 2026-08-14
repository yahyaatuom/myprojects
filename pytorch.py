import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
digit = load_digits()

# Split data with a locked random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    digit.data, digit.target, test_size=0.2, random_state=42
)

# Initialize and train the SVM model
model = SVC(kernel="rbf", C=1.0)
model.fit(X_train, y_train)

# Check model accuracy
accuracy = model.score(X_test, y_test)
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

# Generate predictions for analysis
y_pred = model.predict(X_test)
misclassified_indices = np.where(y_pred != y_test)[0]

# --- UNCOMMENT THIS BLOCK TO VISUALIZE THE SPECIFIC MISTAKES ---
# plt.figure(figsize=(10, 3))
# for i, index in enumerate(misclassified_indices[:4]):
#     plt.subplot(1, 4, i + 1)
#     plt.imshow(X_test[index].reshape(8, 8), cmap='gray')
#     plt.title(f"True: {y_test[index]}\nPred: {y_pred[index]}", color='red')
#     plt.axis("off")
# plt.tight_layout()
# plt.show()
# -------------------------------------------------------------

# Compute and plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digit.target_names)

# Fixed: Changed "blues" to capital "Blues"
disp.plot(cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.show()