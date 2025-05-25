import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Monaco GP race session
session_2024 = fastf1.get_session(2024, "Monaco", "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 2025 Monaco GP Qualifying Data (Only drivers who raced in 2024 Monaco)
# Note: You'll need to replace these with actual 2025 Monaco qualifying times when available
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Yuki Tsunoda", "Alexander Albon", "Esteban Ocon", "Nico Hülkenberg",
               "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.", "Pierre Gasly"],
    "QualifyingTime (s)": [76.200, 76.150, 76.180, 76.100, 76.250,  # These are placeholder times
                           76.120, 76.400, 76.450, 76.380, 76.420,  # Replace with actual 2025 Monaco qualifying
                           76.350, 76.480, 76.300, 76.500]          # times when they become available
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB",
    "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL", "Fernando Alonso": "ALO", "Lance Stroll": "STR",
    "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge qualifying data with sector times from 2024 Monaco
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="inner")

# Define feature set (Qualifying + Sector Times)
X = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
y = merged_data.merge(laps_2024.groupby("Driver")["LapTime (s)"].mean(), left_on="DriverCode", right_index=True)["LapTime (s)"]

# Print number of drivers used in the model
print(f"\n📊 Using data from {len(X)} drivers who participated in both 2024 and 2025 Monaco GP")

# Check if we have enough data
if X.shape[0] < 5:
    print("⚠️ Warning: Limited data available. Consider adding more drivers or using different approach.")

# Train Gradient Boosting Model (adjusted parameters for Monaco's unique characteristics)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(
    n_estimators=150,      # Reduced from 200 due to Monaco's unique nature
    learning_rate=0.08,    # Slightly lower learning rate for better generalization
    max_depth=4,           # Added depth control for Monaco's specific characteristics
    random_state=42
)
model.fit(X_train, y_train)

# Predict race times using 2025 qualifying and 2024 sector data
predicted_race_times = model.predict(X)
merged_data["PredictedRaceTime (s)"] = predicted_race_times

# Rank drivers by predicted race time
merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Monaco GP Winner (Based on 2024 Drivers Only) 🏁\n")
print("=" * 60)
for idx, row in merged_data.iterrows():
    print(f"{merged_data.index.get_loc(idx) + 1:2d}. {row['Driver_x']:<20} - {row['PredictedRaceTime (s)']:.3f}s")

print("\n" + "=" * 60)

# Feature importance analysis (Monaco-specific insights)
feature_names = ["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n🔍 Feature Importance for Monaco GP:")
print(feature_importance_df)

# Evaluate Model
if len(X_test) > 0:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n📊 Model Error (MAE): {mae:.3f} seconds")
else:
    print("\n📊 Model trained on full dataset (no test split due to limited data)")

# Monaco-specific insights
print("\n🏎️ Monaco GP Prediction Insights:")
print("• Monaco is known for qualifying position importance")
print("• Sector 1 (Casino/Mirabeau) and Sector 3 (Swimming Pool/Rascasse) are crucial")
print("• Historical sector performance from 2024 Monaco provides track-specific insights")
print("• Overtaking is extremely difficult - qualifying position heavily influences race result")

print("\n⚠️  Important Notes:")
print("• Replace placeholder qualifying times with actual 2025 Monaco GP qualifying results")
print("• Model predictions are based on 2024 Monaco GP historical data")
print("• Weather conditions and safety cars can significantly impact Monaco race outcomes")
print("• Consider adding more historical Monaco data for improved accuracy")