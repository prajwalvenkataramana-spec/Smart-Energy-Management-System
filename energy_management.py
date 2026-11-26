import pandas as pd
import joblib

# Load trained machine learning model
model = joblib.load("model/energy_model.pkl")

# Load dataset
data = pd.read_csv("dataset/mall_energy.csv")


# Energy prediction function
def predict_energy(temperature, occupancy,
                   ac_usage, lighting_usage):

    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Occupancy": occupancy,
        "AC_Usage": ac_usage,
        "Lighting_Usage": lighting_usage
    }])

    prediction = model.predict(input_data)

    return prediction[0]


# Energy saving recommendations
def energy_recommendations(ac_usage, lighting_usage):

    recommendations = []

    avg_ac = data["AC_Usage"].mean()
    avg_lighting = data["Lighting_Usage"].mean()

    if ac_usage > avg_ac:
        recommendations.append(
            "Reduce unnecessary AC usage."
        )

    if lighting_usage > avg_lighting:
        recommendations.append(
            "Switch off unused lights."
        )

    if not recommendations:
        recommendations.append(
            "Energy usage is within the dataset averages. "
            "Continue monitoring consumption."
        )

    return recommendations


print("SMART MALL ENERGY MANAGEMENT SYSTEM")

temperature = float(input("Enter Temperature: "))
occupancy = float(input("Enter Occupancy: "))
ac_usage = float(input("Enter AC Usage: "))
lighting_usage = float(input("Enter Lighting Usage: "))

# Predict energy consumption
predicted_energy = predict_energy(
    temperature,
    occupancy,
    ac_usage,
    lighting_usage
)

print("\nPredicted Energy Consumption:",
      round(predicted_energy, 2))

# Display recommendations
print("\nENERGY SAVING RECOMMENDATIONS:")

recommendations = energy_recommendations(
    ac_usage,
    lighting_usage
)

for recommendation in recommendations:
    print("-", recommendation)