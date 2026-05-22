"""
Problem:
Gradient Checkpointing.

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

# Implement your function below.
def checkpoint_forward(funcs, input_arr):
    """
    Applies a list of functions in sequence to the input array, simulating gradient checkpointing by not storing intermediates.

    Args:
        funcs (list of callables): List of functions to apply in sequence.
        input_arr (np.ndarray): Input numpy array.

    Returns:
        np.ndarray: The output after applying all functions, same shape as output of last function.
    """
    output = input_arr.astype(float)
    for f in funcs:
        output = f(output)

    return output
