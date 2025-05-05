import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the CSV File ---
file_path = 'road_accident_dataset_small.csv'
df = pd.read_csv(file_path)

print("Raw Data:\n", df.head(), "\n")

# --- Stacked Bar Plot: Fatalities by Road Type and Region ---
stacked = df.groupby(['Region', 'Road Type'])['Number of Fatalities'].sum().unstack(fill_value=0)

stacked.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20c')
plt.title('Total Fatalities by Road Type per Region')
plt.ylabel('Number of Fatalities')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.legend(title='Road Type')
plt.tight_layout()
plt.show()

grouped = df.groupby(['Weather Conditions', 'Region'])['Number of Injuries'].sum().unstack(fill_value=0)

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(grouped, annot=True, fmt=".0f", cmap="YlOrBr")
plt.title("Total Number of Injuries by Weather Condition and Region")
plt.xlabel("Region")
plt.ylabel("Weather Condition")
plt.tight_layout()
plt.show()

# Group by Year and sum the number of injuries
injury_trend = df.groupby('Year')['Number of Injuries'].sum().reset_index()

# Line plot: Number of Injuries by Year
plt.figure(figsize=(10, 6))
plt.plot(injury_trend['Year'], injury_trend['Number of Injuries'],
         label='Injuries', marker='o', color='crimson')

plt.title('Trend Analysis: Number of Injuries Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Number of Injuries')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()