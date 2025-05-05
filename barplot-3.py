import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset if not already loaded
file_path = 'D:/Users/For WEBTOON/PROGRAMMING/Python/elbizan-kmeans/road_accident_dataset_small_clustered.csv'
data = pd.read_csv(file_path)

# Create a severity metric
data['Severity'] = data['Number of Injuries'] + data['Number of Fatalities']

# Identify and handle non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
print("Available columns:", data.columns.tolist())

# Identify potential grouping variables (time-related or categorical numerics)
potential_grouping_cols = [col for col in numeric_columns if col not in
                          ['Cluster', 'Number of Injuries', 'Number of Fatalities', 'Severity']]

if len(potential_grouping_cols) >= 1:
    # Select first potential grouping variable
    grouping_var = potential_grouping_cols[0]
    print(f"Using {grouping_var} as primary grouping variable")

    # Create bins for the grouping variable to make categorical groups
    num_bins = min(5, data[grouping_var].nunique() if data[grouping_var].nunique() < 10 else 5)

    # Create binned column
    data[f'{grouping_var}_bin'] = pd.cut(data[grouping_var], bins=num_bins, labels=[f'Group {i+1}' for i in range(num_bins)])

    # Group by cluster and binned variable
    grouped_data = data.groupby(['Cluster', f'{grouping_var}_bin']).agg({
        'Number of Injuries': 'mean',
        'Number of Fatalities': 'mean',
        'Severity': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Count'}).reset_index()

    # Calculate percentage
    grouped_data['Percentage'] = (grouped_data['Count'] / len(data) * 100).round(1)

    # Create a heatmap visualization to show relationships
    plt.figure(figsize=(14, 10))
    pivot_injuries = pd.pivot_table(grouped_data, values='Number of Injuries',
                                index='Cluster', columns=f'{grouping_var}_bin')

    plt.subplot(2, 1, 1)
    sns.heatmap(pivot_injuries, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'Heatmap of Average Injuries by Cluster and {grouping_var}')

    pivot_fatalities = pd.pivot_table(grouped_data, values='Number of Fatalities',
                                  index='Cluster', columns=f'{grouping_var}_bin')

    plt.subplot(2, 1, 2)
    sns.heatmap(pivot_fatalities, annot=True, cmap='Reds', fmt='.2f')
    plt.title(f'Heatmap of Average Fatalities by Cluster and {grouping_var}')

    plt.tight_layout()
    plt.show()