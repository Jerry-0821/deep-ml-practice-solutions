"""
Problem:
Scale features using standardization and min-max normalization.

Topic:
Machine learning, preprocessing, feature scaling.

Idea:
For each feature column, compute both z-score standardization and min-max
normalization, then round the scaled values.

Key Steps:
1. Compute column-wise means and standard deviations.
2. Standardize each column as (x - mean) / std.
3. Normalize each column as (x - min) / (max - min).

Complexity:
Time: O(n * d), where n is the number of samples and d is the number of features.
Space: O(n * d) for the scaled arrays.
"""

import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    # TODO: Constant columns have std == 0, which can cause division by zero.
    z = (data - mean) / std
    standardized_data = np.round(z, 4)

    x_min = data.min(axis=0)
    x_max = data.max(axis=0)
    # TODO: Constant columns have x_max == x_min, which can cause division by zero.
    min_max = (data - x_min) / (x_max - x_min)
    normalized_data = np.round(min_max, 4)

    return standardized_data.tolist(), normalized_data.tolist()
