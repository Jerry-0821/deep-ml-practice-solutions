"""
Problem:
Compute the F-score for binary classification.

Topic:
Classification metrics.

Idea:
Use the beta-weighted harmonic mean of precision and recall.

Key Steps:
1. Count TP, FP, and FN.
2. Compute precision and recall.
3. Combine them with the beta-weighted F-score formula.

Complexity:
Time: O(n)
Space: O(1)
"""

import numpy as np


def f_score(y_true, y_pred, beta):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f_score = ((1 + (beta ** 2)) * (precision * recall)) / (
        ((beta ** 2) * precision) + recall
    )
    return round(f_score, 3)
