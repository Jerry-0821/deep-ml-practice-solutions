"""
Problem:
Implementing a Simple RNN.

Topic:
Deep learning, recurrent neural networks.

Idea:
Process a sequence step by step, updating the hidden state with tanh
using the current input and previous hidden state.

Key Steps:
1. Convert inputs, weights, and bias to NumPy arrays.
2. Update the hidden state for each sequence element.
3. Return the final hidden state rounded to four decimals.

Complexity:
Time: O(t * h * (d + h)), where t is sequence length.
Space: O(h)
"""

import numpy as np


def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    Wx, Wh, b, h, = np.array(Wx), np.array(Wh), np.array(b), np.array(initial_hidden_state)

    for x in input_sequence:
        x = np.array(x)
        # Combine the input and previous hidden state through matrix-vector products.
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
    final_hidden_state = [round(float(v), 4) for v in h]
    return final_hidden_state
