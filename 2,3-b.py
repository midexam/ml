# K-Means Clustering Algorithm

import numpy as np

# Dataset
X = np.array([2, 4, 10, 12, 3, 20, 30, 11, 25], dtype=float)

# Number of Clusters
k = 2

# Initial Centroids
centroids = np.array([4, 11], dtype=float)

# K-Means Iterations
for iteration in range(1, 6):

    print(f"\n===== Iteration {iteration} =====")

    # Calculate Distances
    distances = np.abs(X[:, np.newaxis] - centroids)

    # Assign Clusters
    labels = np.argmin(distances, axis=1)

    print("\nPoint\tDist to C0\tDist to C1\tCluster")

    for i in range(len(X)):
        print(
            f"{X[i]:.1f}\t"
            f"{distances[i][0]:.2f}\t\t"
            f"{distances[i][1]:.2f}\t\t"
            f"{labels[i]}"
        )

    # Compute New Centroids
    new_centroids = np.array([
        X[labels == i].mean() for i in range(k)
    ])

    print("\nOld Centroids:", centroids)
    print("New Centroids:", new_centroids)

    # Stop if Converged
    if np.allclose(centroids, new_centroids):
        print("\nDone!")
        break

    centroids = new_centroids

# Predict New Point
new_point = 19
distances = np.abs(new_point - centroids)
cluster = np.argmin(distances)

print("\nNew Point :", new_point)
print("Distances :", distances)
print("Predicted Cluster :", cluster)