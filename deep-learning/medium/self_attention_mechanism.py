"""
Problem:
Compute scaled dot-product self-attention from Q, K, and V matrices.

Topic:
Deep learning, transformers, attention mechanisms.

Idea:
Measure token-to-token similarity with QK.T, scale the scores by sqrt(d_k),
normalize each row with softmax, and use the weights to combine V.

Key Steps:
1. Compute scaled attention scores from Q and K.
2. Apply a numerically stable row-wise softmax.
3. Multiply the attention weights by V to produce the output.

Complexity:
Time: O(seq_len^2 * d_k + seq_len^2 * d_v).
Space: O(seq_len^2) for the attention score and weight matrices.
"""

import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k).
        K: Key matrix of shape (seq_len, d_k).
        V: Value matrix of shape (seq_len, d_v).

    Returns:
        Attention output of shape (seq_len, d_v).
    """
    d_k = K.shape[1]

    # Scale by sqrt(d_k) so large dot products do not dominate the softmax.
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # Subtract the row-wise maximum before exponentiation for numerical stability.
    softmax1 = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    softmax = softmax1 / np.sum(softmax1, axis=1, keepdims=True)

    return np.dot(softmax, V)
