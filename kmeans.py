import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
digits = load_digits()
X = digits.data

# Try different numbers of clusters
k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for k in k_values:

    # Create K-Means model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Fit the model and predict cluster labels
    labels = kmeans.fit_predict(X)

    # Calculate evaluation metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)

    # Display results
    print("Number of clusters:", k)
    print("Inertia:", inertia)
    print("Silhouette Score:", silhouette)
    print("-------------------------")