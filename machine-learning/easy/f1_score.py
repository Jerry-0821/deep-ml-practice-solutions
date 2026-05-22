"""
Problem:
Calculate F1 score from predicted and true labels.

Topic:
Classification metrics.

Idea:
Compute precision and recall from binary predictions, then return their
harmonic mean.

Key Steps:
1. Count TP, FP, and FN.
2. Compute precision and recall with zero guards.
3. Return the F1 score rounded to 3 decimal places.

Complexity:
Time: O(n)
Space: O(n)
"""

import numpy as np
def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score based on true and predicted labels.

    Args:
        y_true (list): True labels (ground truth).
        y_pred (list): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if (TP + FP) != 0:
        precision = (TP) / (TP + FP)
    else:
        precision = 0

    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if (precision + recall) != 0:
        f1 = (2 * ((precision * recall) / (precision + recall)))

    else:
        f1 = 0

    return round(f1,3)
