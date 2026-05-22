"""
Problem:
Implement Batch Normalization for BCHW Input.

Topic:
Deep learning, normalization layers, NumPy broadcasting.

Idea:
Normalize BCHW tensors per channel, then apply learnable scale and
shift parameters.

Key Steps:
1. During training, compute channel statistics across batch, height, and width.
2. Update running mean and variance with momentum.
3. During inference, normalize with the running statistics.

Complexity:
Time: O(B * C * H * W)
Space: O(B * C * H * W)
"""

import numpy as np


def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # For BCHW input, channel statistics are computed across batch and spatial axes.
    if training == True:
        batch_mean = X.mean(axis=(0, 2, 3), keepdims=True)
        batch_var = X.var(axis=(0, 2, 3), keepdims=True)
        X_norm = (X - batch_mean) / np.sqrt(batch_var + epsilon)

        if running_mean is None:
            running_mean = np.zeros_like(batch_mean)
        if running_var is None:
            running_var = np.ones_like(batch_var)

        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var

    else:
        X_norm = (X - running_mean) / np.sqrt(running_var + epsilon)

    y = gamma * X_norm + beta
    return y, running_mean, running_var
