"""
Problem:
Calculate R-squared for regression.

Topic:
Regression metrics.

Idea:
R-squared compares residual error against the total variance in the true
target values.

Key Steps:
1. Compute residual sum of squares.
2. Compute total sum of squares.
3. Return 1 - SSR / SST.

Complexity:
Time: O(n)
Space: O(1)
"""

import numpy as np


def r_squared(y_true, y_pred):
    SSR = np.sum((y_true - y_pred) ** 2)
    SST = np.sum((y_true - y_true.mean()) ** 2)
    R2 = 1 - (SSR / SST)
    return R2
