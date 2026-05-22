"""
Problem:
Implement K-Nearest Neighbors.

Topic:
Machine learning implementation.

Idea:
Implement the requested computation directly while preserving the submitted Deep-ML solution logic.

Key Steps:
1. Prepare the input values in the expected shape or type.
2. Apply the core formula or update rule from the solution.
3. Return the computed result.

Complexity:
Time: O(n), where n is the number of processed values.
Space: O(n) for returned values or intermediate arrays.
"""

import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point

    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return

    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    distance = []
    query_point = np.array(query_point)
    for i, point in enumerate(points):
        point = np.array(point)
        dist = np.sqrt(np.sum((point - query_point)**2))
        distance.append((dist, i, tuple(point)))

    distance.sort(key=lambda x: (x[0], x[1]))
    result_points = []
    for dist, i, point in distance[:k]:
        result_points.append(point)
    return result_points
