import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tkinter import Tk, filedialog

# Load the uploaded file to inspect its contents
file_path = 'road_accident_dataset_small.csv'
data = pd.read_csv(file_path)
print(data)

# Step 3: Select numerical columns for clustering
numerical_columns = [
    'Number of Injuries',
    'Number of Fatalities',
    'Medical Cost',
    'Traffic Volume',
    'Economic Loss'
]
numerical_data = data[numerical_columns]

# Step 4: Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(numerical_data)

# Step 5: Determine optimal clusters using the Elbow Method
inertia = []
cluster_range = range(1, 11)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(cluster_range)
plt.grid()
plt.show()

# Step 6: Perform K-means clustering with optimal clusters (set to 3 as an example)
optimal_clusters = 5  # Adjust this based on the elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

# Step 7: Add cluster labels to the original data
data['Cluster'] = clusters

# Step 8: Visualize clusters using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_data)

plt.figure(figsize=(10, 8))
for cluster in range(optimal_clusters):
    cluster_points = reduced_data[clusters == cluster]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {cluster}",
        alpha=0.6
    )
plt.title("Clusters Visualized in 2D (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

# Step 9: Save the results to a new CSV file
output_path = file_path.replace(".csv", "_clustered.csv")
data.to_csv(output_path, index=False)
print(f"Clustered data saved to: {output_path}")