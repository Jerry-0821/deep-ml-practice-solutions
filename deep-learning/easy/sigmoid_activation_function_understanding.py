"""
Problem:
Sigmoid Activation Function Understanding.

Topic:
Deep learning, activation functions.

Idea:
Use the sigmoid formula 1 / (1 + exp(-z)) to map a scalar input into
the range (0, 1).

Key Steps:
1. Compute exp(-z).
2. Add 1 to the exponential term.
3. Return the reciprocal.

Complexity:
Time: O(1)
Space: O(1)
"""

import math


def sigmoid(z: float) -> float:
    result = 1 / (1 + math.exp(-z))
    return result
