"""
Problem:
Implement Early Stopping Based on Validation Loss.

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

from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_loss = float('inf')
    best_epoch = 0
    np_improv = 0

    for epoch, loss in enumerate(val_losses):
        if best_loss - loss > min_delta:
            best_loss = loss
            best_epoch = epoch
            no_improv = 0

        else:
            no_improv += 1

        if no_improv >= patience:
            return (epoch, best_epoch)

    return (len(val_losses) -1 , best_epoch)
