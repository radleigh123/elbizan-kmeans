import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

file_path = 'D:/Users/For WEBTOON/PROGRAMMING/Python/elbizan-kmeans/road_accident_dataset_small.csv'
data = pd.read_csv(file_path)

# Getting all numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
numerical_data = data[numerical_columns]

# Normalizing the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(numerical_data)

# Determine optimal clusters using the Elbow Method
inertia = []
cluster_range = range(1, 11)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(cluster_range)
plt.grid()
plt.show()

# PCA 2D Scatter Plot
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

data['Cluster'] = clusters

pca = PCA(n_components=0.95)
reduced_data = pca.fit_transform(normalized_data)

plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    plt.scatter(
        reduced_data[clusters == cluster, 0],
        reduced_data[clusters == cluster, 1],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.title("Clusters Visualized in 2D (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()



export_path = file_path.replace(".csv", "_clustered.csv")
data.to_csv(export_path, index=False)
print(f"Clustered data saved to: {export_path}")
