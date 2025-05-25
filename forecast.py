import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/interpolated_plastic_timeseries.csv")

# Aggregate globally or by continent
agg_df = df.groupby('Year')['Plastic_to_River_tons'].sum().reset_index()

# Rename columns for Prophet
agg_df = agg_df.rename(columns={"Year": "ds", "Plastic_to_River_tons": "y"})
agg_df['ds'] = pd.to_datetime(agg_df['ds'], format="%Y")

# Initialize and fit model
model = Prophet()
model.fit(agg_df)

# Create future dataframe
future = model.make_future_dataframe(periods=20, freq='Y')  # Forecast until 2080

# Predict
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title("Global Plastic Waste Rivers Forecast (2015-2080)")
plt.xlabel("Year")
plt.ylabel("Plastic Waste (Tons)")
plt.show()

# Save forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("data/plastic_forecast_2015_2060.csv", index=False)
print("\n✅ Forecast saved!")