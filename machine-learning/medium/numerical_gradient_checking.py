"""
Problem:
Check analytical gradients with centered finite differences.

Topic:
Machine learning, optimization, gradient checking.

Idea:
Approximate each gradient entry by perturbing one element of x at a time, then
compare the numerical gradient with the analytical gradient using relative error.

Key Steps:
1. Create a zero array with the same shape as x for the numerical gradient.
2. Perturb each element by +/- epsilon and apply the centered difference formula.
3. Compute relative error using the norms of both gradients.

Complexity:
Time: O(x.size * cost(f)).
Space: O(x.size) for the numerical gradient and perturbed copies.
"""

import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """

    #create a matrix first
    numercial_grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus.flat[i] += epsilon
        x_minus.flat[i] -= epsilon

        numercial_grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    norm_diff = np.linalg.norm(numercial_grad - analytical_grad)
    norm_sum = np.linalg.norm(numercial_grad) + np.linalg.norm(analytical_grad)

    relative_error = 0.0 if norm_sum == 0 else norm_diff / norm_sum

    return numercial_grad, relative_error
