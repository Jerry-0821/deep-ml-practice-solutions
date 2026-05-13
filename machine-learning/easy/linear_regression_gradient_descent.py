"""
Problem:
Learn linear regression coefficients with batch gradient descent.

Topic:
Machine learning, linear regression, gradient descent.

Idea:
Start with zero weights and repeatedly move them opposite the gradient of
the mean squared error loss.

Key Steps:
1. Initialize theta as a zero column vector.
2. Compute predictions and errors for all samples.
3. Update theta using the batch gradient descent rule.

Complexity:
Time: O(iterations * m * n), where m is the number of samples and n is the number of features.
Space: O(n + m) for weights, predictions, and errors.
"""

import numpy as np


def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n), with the first column used for the intercept.
        y: Target vector of shape (m,).
        alpha: Learning rate.
        iterations: Number of gradient descent iterations.

    Returns:
        Learned weights as a 1D array of shape (n,).
    """
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for i in range(iterations):
        y_pred = X @ theta
        error = y_pred - y
        gradient = (X.T @ error) / m
        theta = theta - alpha * gradient

    return theta.flatten()
