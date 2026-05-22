"""
Problem:
Generate a confusion matrix for binary classification.

Topic:
Classification metrics.

Idea:
Count true positives, false negatives, false positives, and true negatives
from pairs of true and predicted labels.

Key Steps:
1. Iterate over (y_true, y_pred) pairs.
2. Update TP, FN, FP, or TN based on each pair.
3. Return the matrix [[TP, FN], [FP, TN]].

Complexity:
Time: O(n)
Space: O(1)
"""


def confusion_matrix(data):
    TP, FN, FP, TN = 0, 0, 0, 0

    for y_true, y_pred in data:
        if y_true == 1 and y_pred == 1:
            TP += 1
        elif y_true == 1 and y_pred == 0:
            FN += 1
        elif y_true == 0 and y_pred == 1:
            FP += 1
        else:
            TN += 1

    return [[TP, FN], [FP, TN]]
