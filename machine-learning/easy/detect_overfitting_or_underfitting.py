"""
Problem:
Detect Overfitting or Underfitting.

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

def model_fit_quality(training_accuracy, test_accuracy):
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: int, one of '1', '-1', or '0'.
    """
    if training_accuracy - test_accuracy > 0.2:
        return 1
    elif training_accuracy < 0.7 and test_accuracy < 0.7:
        return -1
    else:
        return 0
