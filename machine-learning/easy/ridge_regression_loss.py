"""
Problem:
Compute ridge regression loss.

Topic:
Ridge regression.

Idea:
Combine mean squared error with an L2 penalty on the weight vector.

Key Steps:
1. Predict target values with X @ w.
2. Compute mean squared error.
3. Add alpha times the squared L2 norm of the weights.

Complexity:
Time: O(n d)
Space: O(n)
"""

import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # shape shows [4,2], we get 4 as the n
    n = X.shape[0]

    y_pred = X @ w
    mse = np.sum((y_pred - y_true)**2) / n
    loss = mse + alpha * (np.sum(w ** 2))
    return loss
