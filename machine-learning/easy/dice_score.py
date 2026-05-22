"""
Problem:
Calculate Dice score for binary classification.

Topic:
Classification metrics.

Idea:
Dice score is twice the positive intersection divided by the total number
of positive labels and predictions.

Key Steps:
1. Count the positive intersection.
2. Count positives in true and predicted labels.
3. Return 2 * intersection / total, with a zero guard.

Complexity:
Time: O(n)
Space: O(1)
"""

import numpy as np

def dice_score(y_true, y_pred):
    # Write your code here
    intersection = np.sum((y_pred == 1) & (y_true == 1))
    total = np.sum(y_true == 1) + np.sum(y_pred == 1)

    if total == 0:
        return 0.0

    res = (2 * intersection) / (total)
    return round(res, 3)
