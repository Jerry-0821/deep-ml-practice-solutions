"""
Problem:
Compute precision for binary classification.

Topic:
Classification metrics.

Idea:
Precision measures how many predicted positives are true positives.

Key Steps:
1. Count true positives.
2. Count false positives.
3. Return TP / (TP + FP).

Complexity:
Time: O(n)
Space: O(1)
"""

import numpy as np


def precision(y_true, y_pred):
    # NumPy sums boolean values as 1 for True and 0 for False.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)
