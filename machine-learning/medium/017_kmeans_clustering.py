"""
Problem:
Implement K-Means clustering using the provided initial centroids.

Topic:
Machine learning, clustering, NumPy vectorization.

Idea:
Repeatedly assign each point to its nearest centroid, then update each
centroid to the mean of the points assigned to that cluster.

Key Steps:
1. Convert points and centroids into NumPy arrays.
2. Compute all point-to-centroid distances with broadcasting.
3. Update each centroid from its assigned points, keeping old centroids for empty clusters.

Complexity:
Time: O(max_iterations * n * k * d), where d is the point dimension.
Space: O(n * k) for the distance matrix.
"""

import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    centroids = np.array(initial_centroids)
    points = np.array(points)

    for _ in range(max_iterations):
        dists = np.linalg.norm(points[:,None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = []
        for i in range(k):
            if np.any(labels == i):
                new_centroids.append(points[labels==i].mean(axis=0))
            else:
                new_centroids.append(centroids[i])
        new_centroids = np.array(new_centroids)

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return [tuple(round(v, 4) for v in c) for c in centroids]
