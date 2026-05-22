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
    """
    Calculate the recall metric for binary classification.

    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)

    Returns:
        Recall value as a float
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if (TP + FN) == 0:
        return 0.0
    return float(TP/(TP+FN))
