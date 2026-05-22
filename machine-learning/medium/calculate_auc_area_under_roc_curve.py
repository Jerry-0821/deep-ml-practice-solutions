"""
Problem:
Calculate AUC (Area Under ROC Curve).

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

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).

    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores

    Returns:
        AUC value as a float
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)


    threshold = [np.inf] + sorted(np.unique(y_scores), reverse=True)
    tpr = []
    fpr = []
    for thresh in threshold:
        y_pred = (y_scores >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TPR = TP / P if P > 0 else 0.0
        FPR = FP / N if N > 0 else 0.0
        tpr.append(TPR)
        fpr.append(FPR)

    auc = 0.0
    for i in range(1, len(fpr)):
        width = fpr[i] - fpr[i - 1]
        height = (tpr[i] + tpr[i - 1]) / 2
        auc += width * height

    return auc
