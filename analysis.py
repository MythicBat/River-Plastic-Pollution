import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_river_plastic_data.csv")

# Set plot style
sns.set(style="whitegrid")

# 1. Top 10 rivers by Plastic Waste (2015)
top_2015 = df.nlargest(10, 'Plastic_to_River_2015_tons')[['River_Name', 'Country', 'Plastic_to_River_2015_tons']]
print("\nTop 10 Rivers in 2015 by Plastic Waste")
print(top_2015)

# 2. Top 10 rivers by Plastic Waste (2060)
top_2060 = df.nlargest(10, 'Plastic_to_River_2060_tons')[['River_Name', 'Country', 'Plastic_to_River_2060_tons']]
print("\nTop 10 Rivers in 2060 by Projected Plastic Waste")
print(top_2060)

# Bar plot comparison
plt.figure(figsize=(10,6))
sns.barplot(data=top_2015, x='Plastic_to_River_2015_tons', y='River_Name', color='blue', label='2015')
plt.title('Top 10 Rivers by Plastic Waste in 2015')
plt.xlabel('Plastic Waste (Tons)')
plt.ylabel('River')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(data=top_2060, x='Plastic_to_River_2060_tons', y='River_Name', color='red', label='2060')
plt.title('Top 10 Rivers by Projected Plastic Waste in 2060')
plt.xlabel('Plastic Waste (2060)')
plt.ylabel('River')
plt.tight_layout()
plt.show()

# 3. Continent-wise waste comparison
continent_group = df.groupby('Continent')[['Plastic_to_River_2015_tons', 'Plastic_to_River_2060_tons']].sum().reset_index()

# Bar chart: continent comparison
plt.figure(figsize=(12,6))
continent_group.plot(x='Continent', kind='bar', figsize=(12,6))
plt.title('Plastic Waste to Rivers by Continent (2015 vs 2060)')
plt.xlabel('Continent')
plt.ylabel('Total Plastic Waste (Tons)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()