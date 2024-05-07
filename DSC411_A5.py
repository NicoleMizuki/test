import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import *
import warnings

warnings.filterwarnings("ignore")

# Import the dataset
df = pd.read_csv('sampleKMeansData (1).csv')

# Normalize the features and encode non-numeric variables
scaler = StandardScaler()
encoder = LabelEncoder()
for feature in df.columns:
    if is_numeric_dtype(df[feature]):
        df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
    else:
        df[feature] = encoder.fit_transform(df[feature].values.reshape(-1, 1))

vals = df.values


# Define a function to calculate the Silhouette Coefficient (Silhouette_score) for each data point
def silhouette_score(X, labels):
    n_clusters = len(np.unique(labels))
    A = np.zeros_like(labels, dtype=float)
    B = np.zeros_like(labels, dtype=float)

    for i in range(n_clusters):
        value = labels == i
        cluster_size = np.sum(value)
        if cluster_size == 0:
            continue

        A[value] = pairwise_distances(X[value], metric='euclidean').sum(axis=1) / (cluster_size - 1)

        for j in range(n_clusters):
            if i == j:
                continue
            value = labels == j
            if np.sum(value) == 0:
                continue

            B[value] += pairwise_distances(X[labels == i], X[value], metric='euclidean').sum(axis=0) / np.sum(value)

    sil_score = np.zeros_like(labels, dtype=float)
    for i in range(len(labels)):
        a_i = A[i]
        b_i = B[i] / np.sum(labels == labels[i])
        sil_score[i] = (b_i - a_i) / max(a_i, b_i)
    return sil_score


# Define a function to calculate the average Silhouette Coefficient for a given number of clusters
def avg_silhouette_score(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    return np.mean(sil_score)


# Loop over a range of values for K, and calculate the average Silhouette Coefficient for each value
Ks = range(2, 10)
avg_sil_score = []
for K in Ks:
    avg_sil_coef = avg_silhouette_score(vals, K)
    avg_sil_score.append(avg_sil_coef)

# Store the optimal value of K
optimal_K = np.argmax(avg_sil_score) + 2  # Add 2 to offset the range of Ks

#### Bisecting K-Means Implementation ####

# Keep track of clusters
clusters = 1
# Keep track of out centroids to use for final k-means
init_centroids = []
# loop through algorithm until we arrive at our optimal k value
while clusters <= optimal_K:
    km = KMeans(n_clusters=2)
    km.fit(vals.reshape(-1, 9))

    # Keep track of our centroids for each iteration
    centroids = km.cluster_centers_
    labels = km.labels_

    # Calculate SSE for each cluster
    SSE = [0, 0]
    for data, label in zip(vals, labels):
        SSE[label] += np.square(data - centroids[label]).sum()

    # Select our cluster based on SSE
    new_cluster = np.argmax(SSE, axis=0)
    init_centroids.append(centroids[new_cluster])
    new_data = vals[labels == new_cluster]
    X = new_data
    clusters += 1

# Final K-Means clustering
init_centroids = np.asarray(init_centroids)
kmeans = KMeans(n_clusters=optimal_K, init=init_centroids)
kmeans.fit(vals.reshape(-1, 9))

# Final Outputs
print("K-Means Clustering with K=" + str(optimal_K))
for cen in kmeans.cluster_centers_:
    print("Centroid: " + str(cen))
print("SSE: " + str(kmeans.inertia_))

# Original dataset with cluster labels
labels = kmeans.fit_predict(vals.reshape(-1, 9))
df['Cluster'] = labels
print(df)

# Create PCA
Sc = StandardScaler()
X = Sc.fit_transform(df)
pca = PCA(3)
pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2', 'PC3'])
pca_data['Cluster'] = labels

# Plot Figure
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('white')
scatter = ax.scatter(pca_data.PC1, pca_data.PC2, pca_data.PC3, c=pca_data.Cluster, s=40)
legend1 = ax.legend(*scatter.legend_elements())
ax.set_xlabel("First Principal Component", fontsize=10)
ax.set_ylabel("Second Principal Component", fontsize=10)
ax.set_zlabel("Third Principal Component", fontsize=10)
ax.add_artist(legend1)
plt.show()
