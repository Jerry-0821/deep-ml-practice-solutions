"""
Problem:
Initialize a single neural network weight matrix with He initialization.

Topic:
Deep learning, neural networks, weight initialization.

Idea:
Choose fan-in or fan-out as the scaling value, then sample from the He normal
or He uniform distribution for a matrix of shape (n_in, n_out).

Key Steps:
1. Set the NumPy random seed for reproducibility.
2. Select fan based on fan_in or fan_out mode.
3. Sample from the requested distribution with He scaling.

Complexity:
Time: O(n_in * n_out).
Space: O(n_in * n_out) for the initialized weight matrix.
"""

import numpy as np


def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.

    Parameters:
        n_in: Number of input units.
        n_out: Number of output units.
        mode: 'fan_in' or 'fan_out'.
        distribution: 'normal' or 'uniform'.
        seed: Random seed for reproducibility.

    Returns:
        numpy array of shape (n_in, n_out) with He-initialized weights.
    """
    np.random.seed(seed)

    if mode == "fan_in":
        fan = n_in
    else:
        fan = n_out

    # Invalid mode values currently fall back to fan_out; zero fan would cause division by zero.
    if distribution == "normal":
        std = np.sqrt(2 / fan)
        return np.random.randn(n_in, n_out) * std
    elif distribution == "uniform":
        bound = np.sqrt(6 / fan)
        return np.random.uniform(-bound, bound, size=(n_in, n_out))

    # Invalid distribution values return None implicitly in the original simple implementation.
