"""
Problem:
GeLU Activation Function.

Topic:
GeLU activation.

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

def GeLU(x: np.ndarray) -> np.ndarray:
    scores = (0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x **3))))
    return np.round(scores, 4)
