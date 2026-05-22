"""
Problem:
Compute Gini impurity for class labels.

Topic:
Decision trees.

Idea:
Start from 1 and subtract the squared probability of each class.

Key Steps:
1. Count the total number of labels.
2. Estimate each class probability.
3. Apply 1 - sum(p_class ** 2).

Complexity:
Time: O(n)
Space: O(k)
"""

import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    n = len(y)
    gini = 1.0
    for cls in set(y):
        p = y.count(cls) / n
        gini -= p **2

    return round(gini,3)
