import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/cleaned_river_plastic_data.csv")

# Simulate time series for each country
time_series = []

for _, row in df.iterrows():
    for year in range(2015, 2061):
        # Linear interpolation between 2015 and 2060
        start = row['Plastic_to_River_2015_tons']
        end = row['Plastic_to_River_2060_tons']
        interpolated = start + ((end - start) / 45) * (year - 2015)

        time_series.append({
            'Country': row['Country'],
            'Continent': row['Continent'],
            'Year': year,
            'Plastic_to_River_tons': interpolated
        })

ts_df = pd.DataFrame(time_series)
ts_df.to_csv("data/interpolated_plastic_timeseries.csv", index=False)

print("\n✅ Simulated annual plastic waste data is saved")