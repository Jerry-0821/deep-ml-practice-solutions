"""
Problem:
Compute QK-Norm attention with row-wise query and key normalization.

Topic:
Deep learning, attention mechanisms, transformer stabilization.

Idea:
L2-normalize each query and key vector before computing attention scores, then
use temperature-scaled softmax to produce attention weights and output values.

Key Steps:
1. Normalize each row of Q and K with an epsilon for zero-norm safety.
2. Compute temperature-scaled scores from the normalized vectors.
3. Apply a stable row-wise softmax and multiply the weights by V.

Complexity:
Time: O(seq_len_q * seq_len_k * d_k + seq_len_q * seq_len_k * d_v).
Space: O(seq_len_q * seq_len_k) for the attention scores and weights.
"""

import numpy as np


def qk_norm_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, temperature: float = 1.0) -> tuple:
    """
    Apply QK-Norm attention: L2-normalize queries and keys before computing
    scaled dot-product attention.

    Args:
        Q: Query matrix, shape (seq_len_q, d_k).
        K: Key matrix, shape (seq_len_k, d_k).
        V: Value matrix, shape (seq_len_k, d_v).
        temperature: Temperature scaling parameter.

    Returns:
        Tuple of (attention_output, attention_weights).
    """
    # keepdims=True keeps norms broadcastable against the original matrices.
    Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    K_norm = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)

    # temperature=0 would cause division by zero; keep behavior simple for this practice solution.
    scores = np.dot(Q_norm, K_norm.T) / temperature

    # Subtract the row-wise maximum before exponentiation for numerical stability.
    softmax1 = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights = softmax1 / np.sum(softmax1, axis=1, keepdims=True)

    return np.dot(weights, V), weights
