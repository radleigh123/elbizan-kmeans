import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset
file_path = 'D:/Users/For WEBTOON/PROGRAMMING/Python/elbizan-kmeans/road_accident_dataset_small_clustered.csv'
data = pd.read_csv(file_path)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
fig.suptitle('Road Accident Cluster Analysis', fontsize=16)

# Plot 1: Enhanced distribution of clusters with percentages
cluster_counts = data['Cluster'].value_counts().sort_index()
total = len(data)
ax1.bar(cluster_counts.index, cluster_counts.values, color=sns.color_palette('tab10', len(cluster_counts)))

# Add count and percentage labels
for i, (count, cluster) in enumerate(zip(cluster_counts.values, cluster_counts.index)):
    percentage = 100 * count / total
    ax1.text(cluster, count + 0.1, f'n: {count}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontweight='bold')

ax1.set_title('Distribution of Accident Clusters')
ax1.set_xlabel('Cluster (Accident Type)')
ax1.set_ylabel('Number of Accidents')
ax1.set_xticks(cluster_counts.index)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Radar chart or feature importance for clusters
# First, identify numerical columns for analysis (excluding Cluster and any ID columns)
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if 'Cluster' in numerical_cols:
    numerical_cols.remove('Cluster')

# Calculate mean values for each feature by cluster
cluster_means = data.groupby('Cluster')[numerical_cols].mean()

# Normalize the means for better visualization
normalized_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

# Create heatmap of feature distributions across clusters
sns.heatmap(normalized_means, cmap='YlGnBu', annot=True, fmt='.2f', ax=ax2)
ax2.set_title('Normalized Feature Distribution Across Clusters')
ax2.set_ylabel('Cluster')
ax2.set_xlabel('Features')

# Add cluster summary table as text
cluster_summary = data.groupby('Cluster').size().reset_index()
cluster_summary.columns = ['Cluster', 'Count']
cluster_summary['Percentage'] = 100 * cluster_summary['Count'] / total

summary_text = "Cluster Summary:\n"
for _, row in cluster_summary.iterrows():
    summary_text += f"Cluster {row['Cluster']}: {row['Count']} accidents ({row['Percentage']:.1f}%)\n"

fig.text(0.13, 0.01, summary_text, fontsize=10, va='top', ha='left')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to make room for text
plt.show()

# Additional analysis - print defining features for each cluster
print("\nKey characteristics of each cluster:")
for cluster in sorted(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_data)} accidents, {100*len(cluster_data)/total:.1f}%):")
    
    # Find the top 3 distinguishing features (highest mean values compared to other clusters)
    cluster_means = data.groupby('Cluster')[numerical_cols].mean()
    distinctive_features = (cluster_means.loc[cluster] / cluster_means.mean()).sort_values(ascending=False).head(3)
    
    for feature, value in distinctive_features.items():
        print(f"- High {feature}: {value:.2f}x the average")
