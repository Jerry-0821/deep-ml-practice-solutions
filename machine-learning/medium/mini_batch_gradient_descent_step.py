"""
Problem:
Perform one mini-batch gradient descent update for linear regression.

Topic:
Machine learning, linear regression, mini-batch gradient descent.

Idea:
Use only the selected batch rows to compute MSE gradients for the weights and
bias, then return the updated parameters in one vector.

Key Steps:
1. Select the mini-batch from X and y using batch_indices.
2. Compute prediction errors for the current weights and bias.
3. Apply one gradient descent update and append the updated bias.

Complexity:
Time: O(m * D), where m is the batch size and D is the number of features.
Space: O(m + D) for the batch errors and gradients.
"""

import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X_b = X[batch_indices]
    y_b = y[batch_indices]
    m = len(batch_indices)

    e = np.dot(X_b, weights) + bias - y_b
    L_b = 2/m * np.dot(X_b.T, e)
    dL_b = 2/m * np.sum(e)

    #update
    weights -= lr * L_b
    bias -= lr *dL_b

    return np.append(weights, bias)
