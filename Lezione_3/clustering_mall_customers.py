from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

# Load the dataset
data = pd.read_csv('Lezione_3/Mall Customer dataset/Mall_Customers.csv')

data = data.drop(['CustomerID', 'Genre', 'Age'], axis=1)

print(" first 5 rows of the dataset:")
print(data.head())

print("Descriptive statistics of the dataset:")
print(data.describe())

print("Data types of the dataset:")
print(data.info())

print("Checking for missing values in the dataset:")
print(data.isnull().sum())

# cambia il nome delle colonne per comodità
data.columns = ['Annual_Income', 'Spending_Score']

print("First 5 rows after renaming columns:")
print(data.head())

# evidenziamo la distanza euclidea tra le prime due istanze
print("Euclidean distance between the first two instances:")

# non so se eliminare gli outliers o meno, per ora li lascio

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

dist = euclidean_distances([data_scaled[0]], [data_scaled[1]])
print("Euclidean distance:", dist)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data_scaled)

print("KMeans cluster centers:")
print(kmeans.cluster_centers_)

#plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('KMeans Clustering of Mall Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# Calculate silhouette score for KMeans
silhouette_kmeans = silhouette_score(data_scaled, kmeans.labels_)
print("Silhouette Score for KMeans:", silhouette_kmeans)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data_scaled)

# Plotting the DBSCAN clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=dbscan.labels_, cmap='viridis', marker='o', edgecolor='k')
plt.title('DBSCAN Clustering of Mall Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.colorbar(label='Cluster Label')
plt.show()

silhouette_dbscan = silhouette_score(data_scaled, dbscan.labels_)
print("Silhouette Score for DBSCAN:", silhouette_dbscan)


# Apply DBSCAN clustering optimized
dbscan_optimized = DBSCAN(eps=0.38, min_samples=5)
dbscan_optimized.fit(data_scaled)

# Plotting the optimized DBSCAN clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=dbscan_optimized.labels_, cmap='viridis', marker='o', edgecolor='k')
plt.title('Optimized DBSCAN Clustering of Mall Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.colorbar(label='Cluster Label')
plt.show()

# Calculate silhouette score for optimized DBSCAN
silhouette_dbscan_optimized = silhouette_score(data_scaled, dbscan_optimized.labels_)
print("Silhouette Score for Optimized DBSCAN:", silhouette_dbscan_optimized)

# differenza tra i due silhouette score
print("Difference in Silhouette Scores between DBSCAN and Optimized DBSCAN:", silhouette_dbscan_optimized - silhouette_dbscan)
