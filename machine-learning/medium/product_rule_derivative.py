"""
Problem:
Product Rule Derivative.

Topic:
Machine learning implementation.

Idea:
Multiply two polynomials first, then compute the derivative coefficients of
the product.

Key Steps:
1. Reverse coefficient order for NumPy polynomial multiplication.
2. Differentiate the resulting product coefficients.
3. Round coefficients and remove trailing zeros.

Complexity:
Time: O(n * m), where n and m are polynomial lengths.
Space: O(n + m) for product and derivative coefficients.
"""

import numpy as np


def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    """
    product = np.polymul(f_coeffs[::-1], g_coeffs[::-1])[::-1]
    derivative = [i * c for i, c in enumerate(product)][1:]
    if not derivative:
        return 0.0

    result = [round(float(c), 4) for c in derivative]
    while len(result) > 1 and result[-1] == 0.0:
        result.pop()
    return result if result else [0.0]
