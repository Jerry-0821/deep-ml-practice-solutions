"""
Problem:
Implement Polynomial Kernel Function.

Topic:
Machine learning implementation.

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

def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> float:
    """
    Compute the polynomial kernel between two vectors.

    Args:
        x: First input vector
        y: Second input vector
        degree: Degree of the polynomial (default: 3)
        gamma: Scaling factor (default: 1.0)
        coef0: Independent term in kernel function (default: 1.0)

    Returns:
        The polynomial kernel value as a float
    """
    return ((gamma * np.dot(x, y)) + coef0) **degree
