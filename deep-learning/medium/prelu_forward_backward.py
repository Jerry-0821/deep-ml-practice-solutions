"""
Problem:
Implement the PReLU activation function.

Topic:
Activation functions.

Idea:
Return the input unchanged for positive values and use a learnable
negative slope for non-positive values.

Key Steps:
1. Compare the input with zero.
2. Keep positive values unchanged.
3. Scale negative values by alpha.

Complexity:
Time: O(1)
Space: O(1)
"""


def prelu(x: float, alpha: float = 0.25) -> float:
    return x if x > 0 else alpha * x
