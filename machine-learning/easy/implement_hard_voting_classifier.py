"""
Problem:
Implement Hard Voting Classifier.

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

def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.

    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j

    Returns:
        List of final predictions using majority vote
    """
    predictions = np.array(predictions)
    n_samples = predictions.shape[1]
    final_predictions = []

    for i in range(n_samples):
        votes = predictions[:, i]
        majority_class = np.argmax(np.bincount(votes))
        final_predictions.append(int(majority_class))

    return final_predictions
