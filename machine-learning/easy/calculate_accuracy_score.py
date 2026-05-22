"""
Problem:
Calculate the classification accuracy from true and predicted labels.

Topic:
Machine learning, model evaluation, classification metrics.

Idea:
Compare labels element by element and divide the number of correct predictions
by the total number of labels.

Key Steps:
1. Build a boolean array showing which predictions match.
2. Sum the correct predictions.
3. Divide by the number of true labels.

Complexity:
Time: O(n), where n is the number of labels.
Space: O(n) for the boolean comparison array.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    correct = y_true == y_pred
    # Empty y_true would cause division by zero.
    auc = np.sum(correct) / len(y_true)

    return auc
