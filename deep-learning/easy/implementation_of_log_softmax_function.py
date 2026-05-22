"""
Problem:
Implementation of Log Softmax Function.

Topic:
Deep learning, activation functions, numerical stability.

Idea:
Compute log-softmax by subtracting the maximum score before applying
the log-sum-exp normalization.

Key Steps:
1. Convert the input scores to a NumPy array.
2. Shift scores by their maximum value.
3. Subtract log(sum(exp(shifted_scores))) from each shifted score.

Complexity:
Time: O(n)
Space: O(n)
"""

import numpy as np


def log_softmax(scores: list) -> np.ndarray:
    scores = np.array(scores)
    max_score = np.max(scores)

    return scores - max_score - np.log(np.sum(np.exp(scores - max_score)))
