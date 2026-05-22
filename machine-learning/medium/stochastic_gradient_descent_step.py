"""
Problem:
Run stochastic gradient descent updates for linear regression.

Topic:
Machine learning, linear regression, stochastic gradient descent.

Idea:
Cycle through one training sample per iteration, compute the per-sample MSE
gradient, and update the weights immediately.

Key Steps:
1. Select sample t % n for each SGD iteration.
2. Compute the prediction error for that single sample.
3. Update the weight vector using the per-sample gradient.

Complexity:
Time: O(n_iter * D), where D is the number of features.
Space: O(D) for the current sample and weight update.
"""

import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n = len(X)

    for t in range(n_iter):
        i = t % n
        X_i = X[i]
        y_i = y[i]

        y_hat = np.dot(X_i.T, weights)
        L = y_hat - y_i
        weights = weights - 2 * learning_rate * (L) * X_i

    return weights.tolist()
