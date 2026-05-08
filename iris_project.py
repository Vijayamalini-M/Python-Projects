import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("iris.csv")

# Drop unnecessary column
if "Id" in data.columns:
    data = data.drop("Id", axis=1)

# Features & Target
X = data.drop("Species", axis=1)
y = data["Species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Sample prediction (no warning ✅)
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(sample)
print("Predicted Flower:", prediction[0])