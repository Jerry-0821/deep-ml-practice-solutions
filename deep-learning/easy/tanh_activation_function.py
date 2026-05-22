"""
Problem:
Implement the tanh activation function.

Topic:
Activation functions.

Idea:
Use the exponential definition of tanh to map an input value into the
range [-1, 1].

Key Steps:
1. Compute exp(x) and exp(-x).
2. Apply (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
3. Return the tanh value.

Complexity:
Time: O(1)
Space: O(1)
"""

import numpy as np


def tanh(x: float) -> float:
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return result
