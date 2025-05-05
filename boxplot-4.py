import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset if not already loaded
file_path = 'D:/Users/For WEBTOON/PROGRAMMING/Python/elbizan-kmeans/road_accident_dataset_small_clustered.csv'
data = pd.read_csv(file_path)

# Boxplot: Traffic Volume per Cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Traffic Volume', data=data, palette='viridis')
plt.title('Traffic Volume per Accident Type (Cluster)')
plt.xlabel('Cluster (Accident Type)')
plt.ylabel('Traffic Volume')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
