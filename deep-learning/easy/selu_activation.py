"""
Problem:
Compute the SELU activation value for a single input.

Topic:
Deep learning, activation functions, self-normalizing networks.

Idea:
Use the SELU piecewise definition: scale x for positive inputs, and scale the
ELU-style exponential branch for non-positive inputs.

Key Steps:
1. Store the standard SELU alpha and scale constants.
2. Return scale * x for positive inputs.
3. Return scale * alpha * (exp(x) - 1) otherwise.

Complexity:
Time: O(1).
Space: O(1).
"""

import math


def selu(x: float) -> float:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.

    Args:
        x: Input value.

    Returns:
        SELU activation value.
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554804

    if x > 0:
        return scale * x
    else:
        return scale * alpha * (math.exp(x) - 1)
