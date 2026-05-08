import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("CarPrice.csv")

# Drop unwanted columns
data = data.drop(['car_ID', 'CarName'], axis=1)

# Convert categorical to numeric
data = pd.get_dummies(data, drop_first=True)

# Remove missing values
data = data.dropna()

# Features & target
X = data.drop("price", axis=1)
y = data["price"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("\n📊 MODEL PERFORMANCE")
print("------------------------")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 3))

print("\n🔍 SAMPLE PREDICTIONS")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]} | Predicted: {round(y_pred[i],2)}")