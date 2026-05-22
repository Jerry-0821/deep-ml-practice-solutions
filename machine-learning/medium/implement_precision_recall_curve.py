"""
Problem:
Implement Precision-Recall Curve.

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


def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.

    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores

    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    thresholds = sorted(set(y_scores), reverse=True)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    total_positives = y_true.sum()
    precision, recall = [], []
    for t in thresholds:
        predicted = y_scores >= t
        tp = (predicted & y_true.astype(bool)).sum()
        pp = predicted.sum()

        # Precision divides true positives by all predicted positives.
        precision.append(float(tp / pp) if pp > 0 else 1.0)

        # Recall divides true positives by all actual positives.
        recall.append(float(tp / total_positives) if total_positives > 0 else 0.0)

    return precision, recall, thresholds
