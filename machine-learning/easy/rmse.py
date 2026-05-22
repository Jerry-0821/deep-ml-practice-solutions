"""
Problem:
Calculate root mean square error.

Topic:
Regression metrics.

Idea:
RMSE is the square root of the mean squared difference between true and
predicted values.

Key Steps:
1. Validate that both inputs are non-empty NumPy arrays with matching shapes.
2. Compute mean squared error.
3. Return the square root rounded to 3 decimal places.

Complexity:
Time: O(n)
Space: O(1)
"""

import numpy as np


def rmse(y_true, y_pred):
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input array must not be empty")

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    MSE = np.mean((y_true - y_pred) ** 2)
    rmse_res = float(np.sqrt(MSE))

    return round(rmse_res, 3)
