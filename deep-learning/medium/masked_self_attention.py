"""
Problem:
Compute masked self-attention for autoregressive sequence modeling.

Topic:
Deep learning, attention mechanisms, sequence modeling.

Idea:
Use scaled dot-product attention, add a causal mask to block future positions,
apply a stable row-wise softmax, and combine the values with the attention weights.

Key Steps:
1. Compute QK.T scores scaled by sqrt(d_k).
2. Add the mask before softmax so blocked positions get zero probability.
3. Multiply the attention probabilities by V to get the final output.

Complexity:
Time: O(seq_len^2 * d_model) for attention scores and output projection.
Space: O(seq_len^2) for the attention score matrix.
"""

import numpy as np


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)


def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    d_k = K.shape[1]

    # Scale QK.T by sqrt(d_k) to keep softmax values numerically stable.
    score = np.dot(Q, K.T) / np.sqrt(d_k)

    # The mask contains 0 for allowed positions and -inf for blocked positions.
    masked_score = score + mask

    # Subtract the row-wise maximum before exp to reduce overflow risk.
    softmax1 = np.exp(masked_score - np.max(masked_score, axis=1, keepdims=True))
    softmax = softmax1 / np.sum(softmax1, axis=1, keepdims=True)

    return np.dot(softmax, V)
