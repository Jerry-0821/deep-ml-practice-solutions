"""
Problem:
Compute recall for binary classification.

Topic:
Classification metrics.

Idea:
Recall measures how many true positives are recovered by the classifier.

Key Steps:
1. Count true positives.
2. Count false negatives.
3. Return TP / (TP + FN), with a zero guard.

Complexity:
Time: O(n)
Space: O(1)
"""

import numpy as np


def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if (tp + fn) == 0:
        return 0.0

    return float(tp / (tp + fn))
