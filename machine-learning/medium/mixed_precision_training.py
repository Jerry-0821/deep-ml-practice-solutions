"""
Problem:
Mixed Precision Training.

Topic:
Optimization.

Idea:
Simulate mixed precision by doing the forward computation in float16, scaling
the loss, and unscaling gradients in float32.

Key Steps:
1. Cast weights, inputs, and targets to float16 for the forward pass.
2. Compute MSE loss and scale it in float32.
3. Convert gradients to float32, guard overflow, and unscale them.

Complexity:
Time: O(n * d) for the forward matrix multiply.
Space: O(n + d) for cast arrays and gradients.
"""

import numpy as np


class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        self.loss_scale = loss_scale

    def forward(self, weights, inputs, targets):
        # Perform forward pass with float16, then return scaled loss as float32.
        weights16 = weights.astype(np.float16)
        inputs16 = inputs.astype(np.float16)
        targets16 = targets.astype(np.float16)

        predictions = inputs16 @ weights16
        mse_loss = np.mean((predictions - targets16) ** 2)
        scaled_loss = np.float32(mse_loss) * np.float32(self.loss_scale)
        return float(scaled_loss)

    def backward(self, gradients):
        # Unscale gradients and guard against overflow.
        grads32 = gradients.astype(np.float32)
        if np.any(np.isinf(grads32)) or np.any(np.isnan(grads32)):
            return np.zeros_like(grads32, dtype=np.float32)

        unscaled = grads32 / np.float32(self.loss_scale)
        return unscaled.astype(np.float32)
