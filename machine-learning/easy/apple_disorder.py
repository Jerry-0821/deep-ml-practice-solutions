"""
Problem:
Measure disorder in apple colors.

Topic:
Entropy.

Idea:
Use Shannon entropy to measure how mixed the color labels are.

Key Steps:
1. Count how often each color appears.
2. Convert counts into probabilities.
3. Sum -p * log2(p) over all colors.

Complexity:
Time: O(n)
Space: O(k)
"""

from collections import Counter
import math


def disorder(apples: list) -> float:
    n = len(apples)
    if n == 0:
        return 0.0

    counts = Counter(apples)
    entropy = 0

    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)

    return entropy
