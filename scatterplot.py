import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_path = 'D:/Users/For WEBTOON/PROGRAMMING/Python/elbizan-kmeans/road_accident_dataset_small_clustered.csv'
data = pd.read_csv(file_path)

numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
numerical_data = data[numerical_columns]

scaler = StandardScaler()
normalized_data = scaler.fit_transform(numerical_data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_data)

optimal_clusters = 10
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

data['Cluster'] = clusters

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