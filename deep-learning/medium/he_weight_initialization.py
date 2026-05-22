"""
Problem:
Initialize neural network weight matrices with He initialization.

Topic:
Deep learning, neural networks, weight initialization.

Idea:
For each adjacent layer pair, use the input layer size as fan-in and sample
weights from the He normal or He uniform distribution.

Key Steps:
1. Set the NumPy random seed once for reproducible matrices.
2. Build one weight matrix for each adjacent pair of layer dimensions.
3. Scale the sampling distribution using the fan-in of the current layer.

Complexity:
Time: O(total_weights), where total_weights is the sum of all matrix sizes.
Space: O(total_weights) for the generated weight matrices.
"""

import numpy as np


def he_initialize(layer_dims: list, method: str = 'normal', seed: int = 42) -> list:
    """
    Initialize weight matrices for a neural network using He initialization.

    Args:
        layer_dims: List of integers representing neurons per layer.
        method: 'normal' or 'uniform' sampling distribution.
        seed: Random seed for reproducibility.

    Returns:
        List of numpy arrays, one weight matrix per adjacent layer pair.
    """
    np.random.seed(seed)
    weights = []

    # A network with L layer dimensions has L - 1 weight matrices.
    for i in range(len(layer_dims) - 1):
        fan_in = layer_dims[i]
        shape = layer_dims[i], layer_dims[i + 1]

        if method == "normal":
            std = np.sqrt(2 / fan_in)
            W = np.random.normal(loc=0, scale=std, size=shape)
        elif method == "uniform":
            limit = np.sqrt(6 / fan_in)
            W = np.random.uniform(low=-limit, high=limit, size=shape)

        # Invalid methods or zero fan-in would fail; keep behavior simple for this practice solution.
        weights.append(W)

    return weights
