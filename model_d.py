# Import necessary libraries
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

# Create cache directory if it doesn't exist
os.makedirs('f1_cache', exist_ok=True)

# Enable caching
fastf1.Cache.enable_cache('f1_cache')

try:
    # Load Fast1 2024 Monaco GP race data
    session_2024 = fastf1.get_session(2024, 8, "R")  # 2024, Race 8 (Monaco), Race session
    session_2024.load()

    # Extract Lap Times
    laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
    laps_2024.dropna(subset=["LapTime"], inplace=True)
    laps_2024["LapTime"] = laps_2024["LapTime"].dt.total_seconds()

    # 2025 Monaco Qualifying Data (simulated)
    qualifying_2025 = pd.DataFrame({
        "Driver": [
            "Max Verstappen", "Charles Leclerc", "Lando Norris", 
            "Carlos Sainz", "Oscar Piastri", "Lewis Hamilton",
            "George Russell", "Fernando Alonso", "Lance Stroll",
            "Sergio Perez", "Daniel Ricciardo", "Yuki Tsunoda",
            "Nico Hulkenberg", "Kevin Magnussen", "Alexander Albon",
            "Logan Sargeant", "Guanyu Zhou", "Valtteri Bottas",
            "Esteban Ocon", "Pierre Gasly"
        ],
        "QualifyingTime (s)": [
            70.500, 70.650, 70.800,  # Top 3
            70.950, 71.100, 71.250,  # 4-6
            71.400, 71.550, 71.700,  # 7-9
            71.850, 72.000, 72.150,  # 10-12
            72.300, 72.450, 72.600,  # 13-15
            72.750, 72.900, 73.050,  # 16-18
            73.200, 73.350           # 19-20
        ]
    })

    # Map full names to FastF1 3-letter codes
    driver_mapping = {
        "Max Verstappen": "VER", "Charles Leclerc": "LEC", "Lando Norris": "NOR",
        "Carlos Sainz": "SAI", "Oscar Piastri": "PIA", "Lewis Hamilton": "HAM",
        "George Russell": "RUS", "Fernando Alonso": "ALO", "Lance Stroll": "STR",
        "Sergio Perez": "PER", "Daniel Ricciardo": "RIC", "Yuki Tsunoda": "TSU",
        "Nico Hulkenberg": "HUL", "Kevin Magnussen": "MAG", "Alexander Albon": "ALB",
        "Logan Sargeant": "SAR", "Guanyu Zhou": "ZHO", "Valtteri Bottas": "BOT",
        "Esteban Ocon": "OCO", "Pierre Gasly": "GAS"
    }
    qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

    # Merge 2025 Qualifying Data with 2024 Race Data
    merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

    if merged_data.empty:
        raise ValueError("No matching data found between qualifying and race data")

    # Use only "QualifyingTime (s)" as a feature
    X = merged_data[["QualifyingTime (s)"]]  # INPUT: What the model sees
    y = merged_data["LapTime"]               # OUTPUT: What the model learns to predict

    # Train Gradient Boosting Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
    model.fit(X_train, y_train)

    # Predict using 2025 qualifying times
    predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
    qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

    # Rank drivers by predicted race time
    qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

    # Evaluate Model
    y_pred = model.predict(X_test)
    print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
    print("\n2025 Monaco GP Predictions (sorted by predicted race time):")
    print(qualifying_2025[["Driver", "QualifyingTime (s)", "PredictedRaceTime (s)"]].to_string(index=False))

except Exception as e:
    print(f"An error occurred: {str(e)}")