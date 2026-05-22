"""
Problem:
Implement a linear kernel function.

Topic:
Kernel methods.

Idea:
The linear kernel is the dot product between two feature vectors.

Key Steps:
1. Take two input vectors.
2. Compute their dot product.
3. Return the scalar similarity score.

Complexity:
Time: O(n)
Space: O(1)
"""

import numpy as np


def kernel_function(x1, x2):
    return np.dot(x1, x2)
