"""
Problem:
Implement Long Short-Term Memory (LSTM) Network.

Topic:
LSTM recurrent neural networks.

Idea:
Run an LSTM forward pass over a sequence while preserving the submitted gate
update equations.

Key Steps:
1. Concatenate the previous hidden state and current input at each timestep.
2. Compute forget, input, candidate, and output gates.
3. Update the cell state, hidden state, and collect sequence outputs.

Complexity:
Time: O(T * H * (H + D)), where T is sequence length, H is hidden size, and D is input size.
Space: O(T * H) for the returned hidden states.
"""

import numpy as np


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases for the four LSTM gates.
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Process a sequence of inputs and return all hidden states, final hidden state,
        and final cell state.
        """
        h_t = initial_hidden_state
        c_t = initial_cell_state
        output = []

        for t in range(len(x)):
            x_t = x[t].reshape(-1, 1)
            combined = np.vstack((h_t, x_t))

            f = self.Wf @ combined + self.bf
            ft = 1 / (1 + np.exp(-f))

            i = self.Wi @ combined + self.bi
            it = 1 / (1 + np.exp(-i))

            c_hat = self.Wc @ combined + self.bc
            c_hat_t = (np.exp(c_hat) - np.exp(-c_hat)) / (np.exp(c_hat) + np.exp(-c_hat))

            c_t = ft * c_t + it * c_hat_t
            o = self.Wo @ combined + self.bo
            ot = 1 / (1 + np.exp(-o))
            tanh_ct = (np.exp(c_t) - np.exp(-c_t)) / (np.exp(c_t) + np.exp(-c_t))
            h_t = ot * tanh_ct

            output.append(h_t.T)
        return output, h_t.T, c_t.T
