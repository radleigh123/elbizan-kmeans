import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset if not already loaded
file_path = 'D:/Users/For WEBTOON/PROGRAMMING/Python/elbizan-kmeans/road_accident_dataset_small_clustered.csv'
data = pd.read_csv(file_path)

# Boxplot of Medical Cost per Cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Medical Cost', data=data, palette='Set3')
plt.title('Economic Burden per Accident Type (Cluster)')
plt.xlabel('Cluster (Accident Type)')
plt.ylabel('Medical Cost')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

