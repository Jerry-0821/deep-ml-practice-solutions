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


def gini_impurity(y):
    n = len(y)
    gini = 1.0

    for cls in set(y):
        p = y.count(cls) / n
        gini -= p ** 2

    return round(gini, 3)
