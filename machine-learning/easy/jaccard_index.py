"""
Problem:
Calculate the Jaccard index for binary classification.

Topic:
Classification metrics.

Idea:
Jaccard index is intersection over union for positive predictions.

Key Steps:
1. Count positions where both true and predicted labels are positive.
2. Count positions where either label is positive.
3. Return intersection / union, with a zero guard.

Complexity:
Time: O(n)
Space: O(1)
"""

import numpy as np

def jaccard_index(y_true, y_pred):
    # Write your code here
    intersection = np.sum((y_true == 1) &(y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    result = intersection / union if union >0 else 0.0
    return round(result, 3)
