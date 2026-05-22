"""
Problem:
Compute linear regression coefficients using the normal equation.

Topic:
Machine learning, linear regression, matrix algebra.

Idea:
Use the closed-form solution theta = (X.T @ X)^(-1) @ X.T @ y to find
the model coefficients.

Key Steps:
1. Convert X and y into NumPy arrays.
2. Compute X.T @ X and invert it.
3. Apply the normal equation and round each coefficient to four decimals.

Complexity:
Time: O(d^3 + n * d^2), where n is the number of samples and d is the number of features.
Space: O(n * d) for the input matrix representation.
"""

import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    x = np.array(X)
    y = np.array(y)

    theta = (np.linalg.inv(x.T @ x)@ x.T @ y)

    return [float(f"{i:.4f}") for i in theta]
