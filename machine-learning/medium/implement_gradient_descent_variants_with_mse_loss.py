"""
Problem:
Implement Gradient Descent Variants with MSE Loss.

Topic:
Gradient descent.

Idea:
Use one shared MSE gradient helper and apply it with batch, stochastic, or
mini-batch update schedules.

Key Steps:
1. Copy the initial weights to avoid mutating the caller's array directly.
2. Compute MSE gradients for the selected data slice.
3. Apply updates according to the requested gradient descent variant.

Complexity:
Time: O(E * n * d), where E is epochs, n is samples, and d is features.
Space: O(d) for the copied weight vector.
"""

import numpy as np


def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method="batch"):
    """
    Perform gradient descent optimization.
    """
    m = len(y)
    weights = weights.copy()

    def gradient(x_b, y_b):
        y_pred = x_b @ weights
        return (2 / len(y_b)) * x_b.T @ (y_pred - y_b)

    for _ in range(n_epochs):
        if method == "batch":
            weights -= learning_rate * gradient(X, y)

        elif method == "stochastic":
            for i in range(m):
                # Slice one row while preserving a 2D matrix shape.
                weights -= learning_rate * gradient(X[i : i + 1], y[i : i + 1])

        elif method == "mini_batch":
            for start in range(0, m, batch_size):
                x_b, y_b = X[start : start + batch_size], y[start : start + batch_size]
                weights -= learning_rate * gradient(x_b, y_b)

    return weights
