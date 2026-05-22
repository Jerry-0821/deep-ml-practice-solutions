"""
Problem:
Compute Confusion Matrix with Normalization.

Topic:
Classification metrics.

Idea:
Build a multiclass confusion matrix and optionally normalize it by true labels,
predicted labels, or all entries.

Key Steps:
1. Count each (true_label, predicted_label) pair in a K x K matrix.
2. Normalize by rows, columns, or total count when requested.
3. Round normalized values and return a nested Python list.

Complexity:
Time: O(n + K^2), where n is samples and K is number of classes.
Space: O(K^2) for the confusion matrix.
"""

import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=None, round_decimals=4):
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Iterable of true labels in [0, K-1]
        y_pred: Iterable of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        list[list[int|float]] confusion matrix
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t,p] += 1

    if normalize is None:
        return cm.tolist()
    if normalize == "true":
        row_sum = np.sum(cm, axis=1, keepdims=True)
        cm = np.where(row_sum ==0, 0.0, cm / row_sum)

    elif normalize == "pred":
        col_sum = np.sum(cm, axis=0, keepdims=True)
        cm = np.where(col_sum == 0, 0.0, cm / col_sum)

    elif normalize == "all":
        total = np.sum(cm)
        cm = np.where(total == 0, 0.0, cm / total)

    return np.round(cm, decimals=round_decimals).tolist()
