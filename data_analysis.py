import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("dataset/mall_energy.csv")

# Display the first 5 rows
print("Mall Energy Dataset:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display statistical details
print("\nEnergy Consumption Details:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Plot energy consumption
plt.plot(data["Energy_Consumption"], marker="o")
plt.title("Shopping Mall Energy Consumption")
plt.xlabel("Data Records")
plt.ylabel("Energy Consumption")
plt.grid(True)
plt.show()