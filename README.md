# F1 Race Outcome Prediction Model 2025

## Project Overview
This project implements a machine learning system designed to predict Formula 1 race outcomes for the 2025 season. By leveraging historical F1 data from the 2024 season, the system trains predictive models to forecast lap times and race rankings based on various factors including qualifying performance, track conditions, and other relevant metrics.

## Core Concept
The system uses 2024 race data to train a predictive model, which is then applied to 2025 qualifying times to forecast race winners and performance metrics. This approach allows for real-time predictions as new qualifying data becomes available.

## Technical Implementation

### Data Pipeline
1. **Data Collection**
   - Utilized FastF1 API to fetch comprehensive historical race data
   - Automated data retrieval for all 2024 season races

2. **Feature Engineering**
   - Converted lap times to standardized seconds format
   - Mapped and normalized driver names for consistency
   - Processed sector times for advanced predictions

3. **Model Training**
   - Implemented Gradient Boosting Regressor from scikit-learn
   - Optimized hyperparameters for best performance
   - Trained on 2024 season data

4. **Prediction System**
   - Applied trained model to 2025 qualifying data
   - Generated real-time predictions for race outcomes
   - Calculated confidence metrics for predictions

5. **Evaluation**
   - Measured model accuracy using Mean Absolute Error (MAE)
   - Implemented cross-validation for robust performance assessment

### File Breakdown
**model_d.py (Basic Model)**
Purpose: Simplest version predicting Monaco Grand Prix results using only qualifying times.
Approach:
- Uses 2024 Monaco GP race data for training
- Single feature: 2025 qualifying times
- Basic Gradient Boosting model 

**model_c.py**
Purpose: Applies the same basic approach without model improvements.
Key Difference:
- Same simple model architecture
- Includes new 2025 drivers in qualifying data

**model_b.py**
Purpose: Enhanced model using sector times but only for drivers who raced in 2024.
Improvements:
- Adds sector time features (Sector1, Sector2, Sector3)
- Filters out new 2025 drivers
- More sophisticated feature engineering

**model_a.py (Full Model)**
Purpose: Most sophisticated version including all drivers and enhanced features.
Advanced Features:
- Sector time analysis
- Handles new 2025 drivers
- More robust data preprocessing
- Enhanced model parameters

### Machine Learning Approach
- **Algorithm**: Gradient Boosting Regressor (ensemble method)
- **Features**:
  - Qualifying times
  - Sector times (in advanced versions)
  - Track-specific variables
  - Historical performance metrics
- **Target**: Race lap times from 2024
- **Validation**: Train/test split with MAE evaluation

## Features
- Lap time prediction based on historical performance data
- Race ranking forecasts using qualifying results and other key factors
- Analysis of driver and team performance trends
- Integration of track-specific variables and conditions
- Machine learning models trained on 2024 season data

## Technical Stack
- Python for data processing and model development
- FastF1 API for data collection
- scikit-learn for machine learning implementation
- Data analysis libraries (pandas, numpy)
- Visualization tools for results analysis

## Possible Improvements for Better Prediction Model
- More Historical Data
- Driver Form and Momentum Model
- Car Perfomance Model
- Monaco Specific Features (Traffic Management, Sector Performances,...)
- Race Conditions (Weather Probability Model, Safety Car Likelihood, Tire Strategy Optimisation)

## Setup and Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/ml-f1-prediction.git
cd ml-f1-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the prediction system
```bash
python src/main.py
```

## Data Sources
- Historical F1 race data from 2024 season (via FastF1 API)
- Qualifying session results
- Track information and conditions
- Driver and team performance metrics

## Project Structure
```
ml-f1-prediction/
├── data/               # Historical F1 data
├── models/            # Trained ML models
│   ├── data/         # Data collection and processing
│   ├── models/       # ML model implementation
│   └── utils/        # Helper functions
├── notebooks/         # Jupyter notebooks for analysis
└── requirements.txt   # Project dependencies
```


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
