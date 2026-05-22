"""
Problem:
Implement ROC Curve Calculation.

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

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class

    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    P = np.sum(y_true == 1) #(TP+FN)
    N = np.sum(y_true == 0) #(FP+TN)

    #sorted(inverse=True)(from high to low)
    thresholds = [np.inf] + sorted(np.unique(y_scores), reverse=True)
    fpr_list = []
    tpr_list = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TPR = TP/P if P > 0 else 0.0
        FPR = FP/N if N > 0 else 0.0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return (fpr_list, tpr_list)
