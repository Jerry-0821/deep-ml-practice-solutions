"""
Problem:
Randomly shuffle features and labels while keeping their row correspondence.

Topic:
Machine learning, data preprocessing, dataset shuffling.

Idea:
Generate one shuffled index order and apply it to both X and y.

Key Steps:
1. Optionally set the NumPy random seed for reproducibility.
2. Generate a random permutation of sample indices.
3. Index both X and y with the same permutation.

Complexity:
Time: O(n), where n is the number of samples.
Space: O(n) for the shuffled index array.
"""

import numpy as np


def shuffle_data(X, y, seed=None):
    # Only fix the random sequence when the caller provides a seed.
    if seed is not None:
        np.random.seed(seed)

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]
