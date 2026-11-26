import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load dataset
data = pd.read_csv("dataset/mall_energy.csv")

# Select input and output columns
X = data[["Temperature", "Occupancy",
          "AC_Usage", "Lighting_Usage"]]

y = data["Energy_Consumption"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict energy consumption
predictions = model.predict(X_test)

# Display results
print("Actual Energy Consumption:")
print(y_test.values)

print("\nPredicted Energy Consumption:")
print(predictions)

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, predictions))

print("\nR2 Score:")
print(r2_score(y_test, predictions))

# Save the trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/energy_model.pkl")

print("\nModel trained and saved successfully!")