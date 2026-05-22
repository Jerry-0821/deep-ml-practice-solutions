"""
Problem:
Compute gated attention by modulating standard attention output with a sigmoid gate.

Topic:
Deep learning, attention mechanisms, gated neural networks.

Idea:
Compute standard scaled dot-product attention from Q, K, and V projections, then
apply an element-wise sigmoid gate computed from the input.

Key Steps:
1. Project X into Q, K, V, and gate inputs.
2. Compute stable scaled dot-product attention.
3. Multiply the attention output by the sigmoid gate and round the result.

Complexity:
Time: O(seq_len^2 * d_k + seq_len^2 * d_v).
Space: O(seq_len^2 + seq_len * d_v) for attention weights and gated output.
"""

import numpy as np


def gated_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_g: np.ndarray
) -> np.ndarray:
    """
    Compute Gated Attention output.

    Args:
        X: Input tensor of shape (seq_len, d_model).
        W_q: Query projection of shape (d_model, d_k).
        W_k: Key projection of shape (d_model, d_k).
        W_v: Value projection of shape (d_model, d_v).
        W_g: Gate projection of shape (d_model, d_v).

    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    dk = K.shape[1]

    scores = np.dot(Q, K.T) / np.sqrt(dk)

    # Subtract the row-wise maximum before softmax for numerical stability.
    scores = scores - np.max(scores, axis=1, keepdims=True)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    Y = np.dot(attention_weights, V)

    # The sigmoid gate is query-dependent because it is computed from X.
    z = np.dot(X, W_g)
    # Very large positive or negative z values can overflow np.exp in the sigmoid.
    G = 1 / (1 + np.exp(-z))
    Y_gated = G * Y

    return np.round(Y_gated, 4)
