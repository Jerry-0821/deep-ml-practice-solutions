"""
Problem:
Linear Learning Rate Decay.

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

def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.

    Args:
        initial_lr: Starting learning rate
        end_lr: Final learning rate
        num_steps: Total number of training steps

    Returns:
        List of learning rates for each step
    """

    if num_steps == 1:
        return [initial_lr]

    decay = (initial_lr - end_lr) / (num_steps - 1)
    lr = [initial_lr - i * decay for i in range(num_steps)]
    return lr
