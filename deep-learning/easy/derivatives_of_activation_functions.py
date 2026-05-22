"""
Problem:
Derivatives of Activation Functions.

Topic:
Activation functions.

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
def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.

    Args:
        x: Input value

    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    sigmoid = 1 / (1 + np.exp(-x))

    return {'sigmoid': sigmoid * (1 - sigmoid),
     'tanh': 1 - np.tanh(x) ** 2,
     'relu': 1.0 if x > 0 else 0.0}
