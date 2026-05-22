"""
Problem:
Implement ReLU Activation Function.

Topic:
Deep learning, activation functions.

Idea:
Return zero for negative inputs and return the original value for
non-negative inputs.

Key Steps:
1. Check whether the input is negative.
2. Return 0.0 for negative values.
3. Otherwise return the original value.

Complexity:
Time: O(1)
Space: O(1)
"""


def relu(z: float) -> float:
    if z < 0:
        return 0.0
    else:
        return z
