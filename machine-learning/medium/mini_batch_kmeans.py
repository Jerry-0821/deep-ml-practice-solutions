"""
Problem:
Mini-Batch K-Means.

Topic:
Clustering.

Idea:
Initialize random centroids, sample mini-batches, assign each batch point to
the nearest centroid, and update centroids with running learning rates.

Key Steps:
1. Select initial centers using a fixed random seed.
2. Repeatedly sample mini-batches without replacement.
3. Update each selected centroid with a count-based learning rate.

Complexity:
Time: O(I * b * k * d), where I is iterations, b is batch size, k is clusters, and d is features.
Space: O(k * d) for centroids.
"""

import numpy as np

def mini_batch_kmeans(X: np.ndarray, k: int, batch_size: int, max_iters: int, seed: int = 42) -> list:
    """
    Perform Mini-Batch K-Means clustering.

    Args:
        X: Data points of shape (n_samples, n_features)
        k: Number of clusters
        batch_size: Size of each mini-batch
        max_iters: Number of iterations
        seed: Random seed for reproducibility

    Returns:
        List of k centroids, each centroid is a list of coordinates
    """

    np.random.seed(seed)
    n_sample, n_feature = X.shape
    #random choice the center point
    random_center = np.random.choice(n_sample, k, replace=False)
    center_point = X[random_center].astype(float)
    v_counts = np.zeros(k, dtype=np.int64)

    #random choice the batch

    for _ in range(max_iters):
        random_batch = np.random.choice(n_sample, batch_size, replace=False)

        batch = X[random_batch]
        assignment = []
        for x in batch:
            #distance
            distances = np.linalg.norm(center_point - x, axis=1)
            # get the nearest distance point
            nearest_distance = np.argmin(distances)
            assignment.append(nearest_distance)

            #update data
        for x, nearest_distance in zip(batch, assignment):
            v_counts[nearest_distance] += 1
            learning_rate = 1.0 / v_counts[nearest_distance]
            center_point[nearest_distance] = (1 - learning_rate) * center_point[nearest_distance] + learning_rate * x



    return [list(np.round(c, 4)) for c in center_point]

#np.round(center_point.tolist(),  4)
