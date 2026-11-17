import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

digits = load_digits()
X = digits.data

k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for k in k_values:

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)

    print("Number of clusters:", k)
    print("Inertia:", inertia)
    print("Silhouette Score:", silhouette)
    print("-------------------------")
