"""
Problem:
Implement Lasso Regression using ISTA.

Topic:
Regression.

Idea:
Implement the requested computation directly while preserving the submitted Deep-ML solution logic.

Key Steps:
1. Prepare the input values in the expected shape or type.
2. Apply the core formula or update rule from the solution.
3. Return the computed result.

Complexity:
Time: O(n), where n is the number of processed values.
Space: O(n) for returned values or intermediate arrays.
"""

import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.

    S(w, lambda) = sign(w) * max(|w| - lambda, 0)

    Args:
        w: Input array
        threshold: Threshold value lambda

    Returns:
        Soft-thresholded array where:
        - Values with |w| > lambda are shrunk toward zero by lambda
        - Values with |w| <= lambda become exactly zero
    """

    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).

    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change

    Returns:
        tuple: (weights, bias)

    Note: The bias term is not regularized.
    """

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        y_pred = X.dot(weights) + bias

        grad_w = -(1 / n_samples) * X.T.dot(y - y_pred)
        grad_b = -(1 / n_samples) * np.sum(y - y_pred)

        w_temp = weights - learning_rate * grad_w
        bias = bias - learning_rate * grad_b

        weights_new = soft_threshold(w_temp, learning_rate * alpha)

        if np.linalg.norm(weights_new - weights) < tol:
            weights = weights_new
            break
        weights = weights_new
    return weights, bias
