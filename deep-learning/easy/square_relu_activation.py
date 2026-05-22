"""
Problem:
Apply the Square ReLU activation function and compute its derivative.

Topic:
Deep learning, activation functions, ReLU variants.

Idea:
Square positive inputs and set non-positive inputs to zero; the derivative is
2x for positive inputs and 0 otherwise.

Key Steps:
1. Use np.where to apply Square ReLU element-wise.
2. Use np.where again for the piecewise derivative.
3. Round both output arrays to four decimal places.

Complexity:
Time: O(n), where n is the number of input elements.
Space: O(n) for the output and derivative arrays.
"""

import numpy as np

def square_relu(x: np.ndarray) -> dict:
    """
    Apply the Square ReLU activation function and compute its derivative.

    Args:
        x: Input numpy array of any shape

    Returns:
        Dictionary with 'output' and 'derivative' as numpy arrays
    """
    output = np.where(x >0, x**2, 0.0)
    derivative = np.where(x > 0, 2*x, 0.0)

    return {"output": np.round(output, 2), "derivative": np.round(derivative, 2)}
