"""
Problem:
Implement the ELU Activation Function.

Topic:
Deep learning, activation functions.

Idea:
Return positive inputs unchanged and use alpha * (exp(x) - 1) for
negative inputs.

Key Steps:
1. Check whether the input is positive.
2. Use x directly for positive values.
3. Use the ELU exponential branch for negative values and round the result.

Complexity:
Time: O(1)
Space: O(1)
"""

import numpy as np
def elu(x: float, alpha: float = 1.0) -> float:
    """
    Compute the ELU activation function.

    Args:
        x (float): Input value
        alpha (float): ELU parameter for negative values (default: 1.0)

    Returns:
        float: ELU activation value
    """
    val = x if x > 0 else alpha*(np.exp(x)-1)
    return round(val,4)
