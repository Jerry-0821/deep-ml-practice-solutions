"""
Problem:
Convert integer class labels into one-hot encoded rows.

Topic:
Machine learning, preprocessing, categorical encoding.

Idea:
Create a zero matrix and use integer indexing to set each label's category
position to 1.

Key Steps:
1. Infer the number of columns if n_col is not provided.
2. Allocate a zero matrix with one row per input value.
3. Use row indices and label values to place the ones.

Complexity:
Time: O(n), where n is the number of labels.
Space: O(n * c), where c is the number of encoded columns.
"""

import numpy as np


def to_categorical(x, n_col=None):
    if n_col is None:
        # Empty input will make np.max fail.
        n_col = np.max(x) + 1

    one_hot = np.zeros((x.shape[0], n_col))
    # Advanced indexing sets one category position per row.
    one_hot[np.arange(x.shape[0]), x] = 1

    return one_hot
