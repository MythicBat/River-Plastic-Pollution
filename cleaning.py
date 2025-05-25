import pandas as pd

# Load the dataset
df = pd.read_csv("data/River_Plastic_Waste_Risk_Scenarios_2015_2060.csv")

# show basic info
print("\n--- Dataset Overview ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())

# Check for missing values
print("\n--- Missing values ---")
print(df.isnull().sum())

# Drop rows with missing target
df = df.dropna(subset=['Plastic_to_River_2060_tons'])

# Feature Engineering
df['Waste_Per_Capita_2015'] = df['Plastic_to_River_2015_tons']/ df['Population_2015']
df['Risk_Density'] = df['Risk_Score_Change'] / df['River_Length_km']

# Risk Level
def categorize_risk(score):
    if score < 1:
        return 'Low'
    elif score < 3:
        return 'Medium'
    else:
        return 'High'

df['Risk_Level'] = df['Risk_Score_Change'].apply(categorize_risk)

# saved cleaned data
df.to_csv("data/cleaned_river_plastic_data.csv", index=False)
print("\n✅ Cleaned data saved")