"""
Problem:
Random Forest Feature Importance.

Topic:
Decision trees.

Idea:
Aggregate impurity decreases for each feature across all trees and normalize
the totals into relative importances.

Key Steps:
1. Accumulate impurity decreases by feature index.
2. Sum all feature importances.
3. Normalize by the total when it is nonzero.

Complexity:
Time: O(s), where s is the total number of split records.
Space: O(f), where f is the number of features.
"""


def random_forest_feature_importance(trees: list, n_features: int) -> list:
    """
    Calculate feature importance from a random forest using mean decrease in impurity.
    """
    importance = [0.0] * n_features
    for tree in trees:
        for split in tree:
            importance[split["feature_index"]] += split["impurity_decrease"]

    total = sum(importance)
    if total == 0:
        return importance

    return [v / total for v in importance]
