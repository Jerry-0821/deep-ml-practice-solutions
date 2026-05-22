"""
Problem:
Softmax Activation Function Implementation.

Topic:
Deep learning, activation functions, numerical stability.

Idea:
Convert a list of scores into probabilities by exponentiating shifted
scores and normalizing by their total.

Key Steps:
1. Subtract the maximum score before exponentiation.
2. Compute exponentials of the shifted scores.
3. Divide each exponential value by the total sum.

Complexity:
Time: O(n)
Space: O(n)
"""

import math


def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exp_score = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_score)
    return [e / sum_exp for e in exp_score]
