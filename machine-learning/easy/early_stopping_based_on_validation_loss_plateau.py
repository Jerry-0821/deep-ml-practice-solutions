"""
Problem:
Early Stopping Based on Validation Loss Plateau.

Topic:
Machine learning implementation.

Idea:
Track the best validation loss and flag epochs where the loss has failed to
improve enough for the configured patience window.

Key Steps:
1. Keep the best validation loss seen so far.
2. Reset the counter when the loss improves by at least min_delta.
3. Mark stopping once the non-improvement counter reaches patience.

Complexity:
Time: O(n), where n is the number of validation losses.
Space: O(n) for the returned stop flags.
"""


def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
    """
    Determine at each epoch whether training should stop based on validation loss.
    """
    result = []
    best_loss = float("inf")
    counter = 0

    for loss in val_losses:
        if loss < best_loss - min_delta:
            best_loss = loss
            counter = 0
        else:
            counter += 1

        result.append(counter >= patience)

    return result
