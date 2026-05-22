"""
Problem:
Leaky ReLU Activation Function.

Topic:
Deep learning, activation functions.

Idea:
Keep positive values unchanged and scale negative values by a small
slope alpha.

Key Steps:
1. Check whether the input is positive.
2. Return the input directly for positive values.
3. Return alpha times the input for non-positive values.

Complexity:
Time: O(1)
Space: O(1)
"""


def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    return z if z > 0 else alpha * z
