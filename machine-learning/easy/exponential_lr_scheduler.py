"""
Problem:
Implement an exponential learning-rate scheduler.

Topic:
Learning-rate scheduling.

Idea:
Decay the initial learning rate by multiplying gamma once per epoch.

Key Steps:
1. Store the initial learning rate.
2. Store the exponential decay factor.
3. Return initial_lr * gamma ** epoch rounded to 4 decimals.

Complexity:
Time: O(1)
Space: O(1)
"""


class ExponentialLRScheduler:
    def __init__(self, initial_lr, gamma):
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        lr = self.initial_lr * (self.gamma ** epoch)
        return round(lr, 4)
