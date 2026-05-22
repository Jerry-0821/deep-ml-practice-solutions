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
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    f_score = ((1+(beta**2)) * (precision * recall)) / ((beta **2 * precision) + recall)
    return round(f_score, 3)
