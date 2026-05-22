"""
Problem:
Implement Layer Normalization for Sequence Data.

Topic:
Normalization.

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

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    """
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    result = gamma * X_norm + beta
    return result
