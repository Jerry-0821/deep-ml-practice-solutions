"""
Problem:
XGBoost Objective Function Calculation.

Topic:
Machine learning implementation.

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

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.

    Args:
        gradients: First-order gradients for each sample
        hessians: Second-order hessians for each sample
        left_indices: Indices of samples going to left child
        right_indices: Indices of samples going to right child
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty

    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """
    G_left = gradients[left_indices].sum()
    G_right = gradients[right_indices].sum()
    H_left = hessians[left_indices].sum()
    H_right = hessians[right_indices].sum()

    left_weight = - G_left / (H_left + lambda_reg)
    right_weight = - G_right / (H_right + lambda_reg)

    gain = (0.5 * (((G_left**2)/ (H_left+lambda_reg)) +
    ((G_right**2) / (H_right+lambda_reg)) -
    ((G_left+G_right)**2/ (H_left + H_right + lambda_reg))
        )) - gamma

    return {'left_weight': round(left_weight,4), 'right_weight': round(right_weight,4), 'gain': round(gain, 4)}
