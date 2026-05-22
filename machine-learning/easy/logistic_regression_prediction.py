"""
Problem:
Predict binary labels with logistic regression.

Topic:
Logistic regression.

Idea:
Apply the sigmoid function to the linear model output, then threshold
probabilities at 0.5.

Key Steps:
1. Compute X @ weights + bias.
2. Clip logits for numerical stability.
3. Convert sigmoid probabilities into binary labels.

Complexity:
Time: O(n d)
Space: O(n)
"""

import numpy as np


def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = X @ weights + bias
    z = np.clip(z, -500, 500)
    prob = 1 / (1 + np.exp(-z))
    pred = (prob >= 0.5).astype(int)
    return pred
