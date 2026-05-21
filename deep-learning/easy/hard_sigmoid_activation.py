"""
Problem:
Apply the Hard Sigmoid activation function to a single input value.

Topic:
Deep learning, activation functions, sigmoid approximations.

Idea:
Use the Keras-style piecewise linear hard sigmoid: return 0 in the lower
saturation region, 1 in the upper saturation region, and 0.2x + 0.5 in between.

Key Steps:
1. Check whether x is at or below -2.5.
2. Check whether x is at or above 2.5.
3. Apply the linear formula for values inside the active region.

Complexity:
Time: O(1).
Space: O(1).
"""


def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function.

    Args:
        x (float): Input value.

    Returns:
        float: The Hard Sigmoid of the input.
    """
    if x <= -2.5:
        return 0
    elif x >= 2.5:
        return 1
    else:
        return 0.2 * x + 0.5
