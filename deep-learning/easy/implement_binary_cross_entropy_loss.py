"""
Problem:
Compute binary cross-entropy loss.

Topic:
Binary cross-entropy.

Idea:
Clip predicted probabilities for numerical stability, then average the
negative log-likelihood across all labels.

Key Steps:
1. Clamp each prediction away from 0 and 1.
2. Add the binary cross-entropy contribution for each example.
3. Return the mean loss rounded to 4 decimal places.

Complexity:
Time: O(n)
Space: O(1)
"""

import math

def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability

    Returns:
        Mean binary cross-entropy loss
    """
    n = len(y_true)
    total = 0.0
    for y, p in zip(y_true, y_pred):
        # p cannot 1 or 0
        # p cannot be 0
        if p < epsilon:
            p = epsilon
        # p cannot be 1, can 0.999999
        elif p > 1 - epsilon:
            p = 1 - epsilon

        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))

    return round(total / n, 4)
