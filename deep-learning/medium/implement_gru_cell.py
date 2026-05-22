"""
Problem:
Implement GRU Cell.

Topic:
GRU recurrent neural networks.

Idea:
Implement the requested computation directly while preserving the submitted Deep-ML solution logic.

Key Steps:
1. Prepare the input values in the expected shape or type.
2. Apply the core formula or update rule from the solution.
3. Return the computed result.

Complexity:
Time: O(n), where n is the number of processed values.
Space: O(n) for returned values or intermediate arrays.
"""

import numpy as np

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.

    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input
        U_z, U_r, U_h: Weight matrices for hidden state
        b_z, b_r, b_h: Bias vectors

    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    z = (np.dot(W_z, x) + np.dot(U_z, h_prev) + b_z)
    update_gate = 1 / (1 + np.exp(-z))

    r = (np.dot(W_r, x) + np.dot(U_r, h_prev) + b_r)
    reset_gate = 1 / (1 + np.exp(-r))

    candidate_h = np.tanh(np.dot(W_h, x) + np.dot(U_h, (reset_gate * h_prev)) + b_h)

    new_h = (1 - update_gate) * h_prev + update_gate * candidate_h

    return new_h
