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
    points_array = np.array(points)

    for _ in range(max_iterations):
        # Broadcasting gives shape (num_points, k, dimensions), then axis=2
        # reduces each coordinate difference into one Euclidean distance.
        distances = np.linalg.norm(points_array[:, None] - centroids[None, :], axis=2)

        # axis=1 selects the nearest centroid for each point.
        labels = np.argmin(distances, axis=1)
        new_centroids = []

        for i in range(k):
            # labels == i marks all points currently assigned to cluster i.
            if np.any(labels == i):
                new_centroids.append(points_array[labels == i].mean(axis=0))
            else:
                # Keep the previous centroid if this cluster has no assigned points.
                new_centroids.append(centroids[i])

        new_centroids = np.array(new_centroids)

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # Return plain Python tuples with each coordinate rounded to four decimals.
    return [tuple(round(v, 4) for v in centroid) for centroid in centroids]
