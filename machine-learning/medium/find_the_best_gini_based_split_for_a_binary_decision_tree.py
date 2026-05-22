"""
Problem:
Find the Best Gini-Based Split for a Binary Decision Tree.

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
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    n, d = X.shape
    best_feat, best_thresh, best_gini = 0, float("inf"), float("inf")
    def gini_index(labels):
        if len(labels) == 0:
            return 0.0
        p = np.mean(labels)
        gini = 1 - (p **2 + (1 - p) ** 2)
        return gini

    for feat in range(d):
        threshold = np.unique(X[:, feat])
        for thresh in threshold:
            left = y[X[:, feat] <= thresh]
            right = y[X[:, feat] > thresh]
            if len(left) == 0 or len(right) == 0:
                continue
            weighted = (len(left) * gini_index(left) + len(right) * gini_index(right)) / n
            if weighted < best_gini:
                best_gini, best_feat, best_thresh = weighted, feat, thresh

    return (best_feat, best_thresh)
