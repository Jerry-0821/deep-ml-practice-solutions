"""
Problem:
Calculate mean absolute error.

Topic:
Regression metrics.

Idea:
MAE is the mean absolute difference between true and predicted values.

Key Steps:
1. Compute elementwise absolute error.
2. Average all errors.
3. Return the result as a float.

Complexity:
Time: O(n)
Space: O(n)
"""

import numpy as np


def mae(y_true, y_pred):
    error = abs(y_true - y_pred)
    mae = float(np.mean(error))

    return mae
