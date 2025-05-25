import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the cleaned data
df = pd.read_csv("data/cleaned_river_plastic_data.csv")

# Define features (X) and target (y)
features = [
    'Population_2015',
    'Urbanization_2015_pct',
    'Policy_Strength_2015',
    'Waste_Collection_Rate_2015',
    'Plastic_to_River_2015_tons',
    'Risk_Score_Change'
]

X = df[features]
y = df['Plastic_to_River_2060_tons']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function to evaluate models
def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation: ")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate(lr, "Linear Regression")

# 2. Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
evaluate(rf, "Random Forest")

# 3. XGBoost
xgb = XGBRegressor(random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
evaluate(xgb, "XGBoost")

# Save Prediction
df['Predicted_2060_Plastic'] = xgb.predict(X)
df.to_csv("data/predicted_river_plastic_data.csv", index=False)
print("\n✅ Predictions are saved")