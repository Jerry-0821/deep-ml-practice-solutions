"""
Problem:
Implement a dropout layer.

Topic:
Regularization.

Idea:
During training, randomly mask activations and scale the remaining values
by the keep probability. During inference, return the input unchanged.

Key Steps:
1. Store the dropout probability and mask.
2. Generate a Bernoulli mask during training.
3. Reuse the same mask to scale gradients in the backward pass.

Complexity:
Time: O(n)
Space: O(n)
"""

import numpy as np


class DropoutLayer:
    def __init__(self, p: float):
        if not (0 <= p < 1):
            raise ValueError("it is wrong")

        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training == False:
            return x

        self.mask = np.random.binomial(1, 1 - self.p, size=x.shape)
        y = (x * self.mask) / (1 - self.p)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise ValueError("it is wrong")

        grad = grad * self.mask / (1 - self.p)
        return grad
