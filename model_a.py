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
avg_lap_times_2024 = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()

# 2025 Monaco GP Qualifying Data (all drivers)
# Note: Replace these placeholder times with actual 2025 Monaco qualifying results
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [76.200, 76.150, 76.180, 76.100, 76.250,  # Placeholder Monaco times
                           76.120, 76.280, 76.300, 76.400, 76.450,  # Replace with actual results
                           76.380, 76.420, 76.350, 76.480, 76.330,  # when available
                           76.500, 76.520, 76.540, 76.560, 76.580]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge qualifying data with sector times (left join to keep all 2025 drivers)
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")

# Smart imputation for new drivers (instead of zero-fill)
# Calculate average sector time ratios from 2024 data
sector_ratios = sector_times_2024[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean()
total_avg_sector_time = sector_ratios.sum()

print("📊 Monaco Sector Analysis from 2024:")
print(f"Sector 1 (Casino/Mirabeau): {sector_ratios['Sector1Time (s)']:.3f}s ({sector_ratios['Sector1Time (s)']/total_avg_sector_time*100:.1f}%)")
print(f"Sector 2 (Portier/Tunnel): {sector_ratios['Sector2Time (s)']:.3f}s ({sector_ratios['Sector2Time (s)']/total_avg_sector_time*100:.1f}%)")
print(f"Sector 3 (Swimming Pool/Rascasse): {sector_ratios['Sector3Time (s)']:.3f}s ({sector_ratios['Sector3Time (s)']/total_avg_sector_time*100:.1f}%)")

# For new drivers, estimate sector times based on their qualifying performance
# relative to the average qualifying time
avg_qual_time_2024 = merged_data[merged_data["Sector1Time (s)"].notna()]["QualifyingTime (s)"].mean()

for idx, row in merged_data.iterrows():
    if pd.isna(row["Sector1Time (s)"]):  # New driver without 2024 data
        qual_ratio = row["QualifyingTime (s)"] / avg_qual_time_2024
        merged_data.loc[idx, "Sector1Time (s)"] = sector_ratios["Sector1Time (s)"] * qual_ratio
        merged_data.loc[idx, "Sector2Time (s)"] = sector_ratios["Sector2Time (s)"] * qual_ratio
        merged_data.loc[idx, "Sector3Time (s)"] = sector_ratios["Sector3Time (s)"] * qual_ratio
        print(f"🆕 Estimated sector times for {row['Driver_x']} (qualifying ratio: {qual_ratio:.3f})")

# Now create training data using only drivers with actual 2024 race data
training_data = merged_data.merge(avg_lap_times_2024, left_on="DriverCode", right_on="Driver", how="inner")

# Features and target for training (only drivers with historical data)
X_train_full = training_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
y_train_full = training_data["LapTime (s)"]

print(f"\n🏗️ Training Model on {len(X_train_full)} drivers with 2024 Monaco data")

# Train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Train Gradient Boosting Model (Monaco-optimized parameters)
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Predict for ALL 2025 drivers (including new ones with estimated sector times)
X_predict = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
predicted_race_times = model.predict(X_predict)
qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Monaco GP Results (All Drivers) 🏁")
print("=" * 70)
for idx, row in qualifying_2025.iterrows():
    position = qualifying_2025.index.get_loc(idx) + 1
    is_new_driver = row["DriverCode"] not in sector_times_2024["Driver"].values
    marker = "🆕" if is_new_driver else "  "
    print(f"{position:2d}. {marker} {row['Driver']:<22} - {row['PredictedRaceTime (s)']:.3f}s")

print("=" * 70)
print("🆕 = New driver (estimated sector times)")

# Feature importance analysis
feature_names = ["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n🔍 Feature Importance for Monaco GP:")
for _, row in feature_importance_df.iterrows():
    print(f"{row['Feature']:<20}: {row['Importance']:.3f}")

# Evaluate Model (on drivers with historical data)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n📊 Model Error (MAE): {mae:.3f} seconds")
print(f"📊 Training Size: {len(X_train)} drivers")
print(f"📊 Prediction Coverage: {len(qualifying_2025)} drivers")

# Monaco-specific insights
print("\n🏎️ Monaco GP Prediction Insights:")
print("• Qualifying position is crucial at Monaco - very limited overtaking opportunities")
print("• Sector 1 (Casino Square) and Sector 3 (Swimming Pool complex) are typically most challenging")
print("• New drivers' sector times estimated based on qualifying performance relative to 2024 field")
print("• Weather conditions and safety car deployments can dramatically change race outcomes")

print("\n⚠️  Model Limitations:")
print("• New drivers' predictions based on estimated sector times (less reliable)")
print("• Single season (2024) training data - consider adding 2022-2023 Monaco data")
print("• Does not account for tire strategy, fuel loads, or race incidents")
print("• Replace placeholder qualifying times with actual 2025 Monaco GP results")