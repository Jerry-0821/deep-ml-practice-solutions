"""
Problem:
Generate polynomial feature combinations and sort each sample's expanded features.

Topic:
Machine learning, feature engineering, polynomial features.

Idea:
Build all combinations with replacement up to the requested degree, multiply
the selected feature values for each term, and sort each row.

Key Steps:
1. Iterate over each input sample.
2. Generate polynomial feature index combinations for each degree.
3. Multiply the selected values, sort the row, and collect the result.

Complexity:
Time: O(n * c * degree), where c is the number of generated combinations.
Space: O(n * c) for the expanded feature matrix.
"""

import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):

    n_sample, n_feature = X.shape
    result = []

    for i in range(n_sample):
        row  = X[i]
        feature = []
        for d in range(degree + 1):
            for comb in combinations_with_replacement(range(n_feature), d):

                val = 1.0
                for idx in comb:
                    val *= row[idx]

                feature.append(val)
        feature.sort()
        result.append(feature)

    return np.array(result)
