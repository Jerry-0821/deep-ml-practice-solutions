"""
Problem:
Single Neuron with Backpropagation.

Topic:
Deep learning, sigmoid neuron, gradient descent.

Idea:
Train one sigmoid neuron by computing predictions, measuring mean
squared error, and updating weights with backpropagation.

Key Steps:
1. Compute sigmoid predictions from X @ w + b.
2. Track the mean squared error at each epoch.
3. Use the chain rule to update weights and bias.

Complexity:
Time: O(epochs * n * d), where d is the number of features.
Space: O(n + d)
"""

import numpy as np


def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    X = np.array(features)
    y = np.array(labels)
    w = np.array(initial_weights, dtype=float)
    b = float(initial_bias)
    mse_values = []

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for _ in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        error = y_hat - y
        mse = np.mean(error ** 2)
        mse_values.append(round(float(mse), 4))

        # Backpropagate through the sigmoid and mean squared error.
        dL_dz = 2 * error * y_hat * (1 - y_hat)
        dL_dw = np.dot(X.T, dL_dz) / len(y)
        dL_db = np.mean(dL_dz)

        w -= learning_rate * dL_dw
        b -= learning_rate * dL_db

    updated_weights = np.round(w, 4).tolist()
    updated_bias = round(b, 4)
    return updated_weights, updated_bias, mse_values
