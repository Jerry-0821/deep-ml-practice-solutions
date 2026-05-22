"""
Problem:
Taylor Series Approximation.

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

import numpy as np
from math import factorial

def taylor_approximation(func_name: str, x: float, n_terms: int) -> float:
    """
    Compute Taylor series approximation for common functions.

    Args:
        func_name: Name of function ('exp', 'sin', 'cos')
        x: Point at which to evaluate
        n_terms: Number of terms in the series

    Returns:
        Taylor series approximation rounded to 6 decimal places
    """
    result = []

    if func_name == "exp":
        for n in range(n_terms):
            a = x ** n / factorial(n)
            result.append(a)

    if func_name == "sin":
        for n in range(n_terms):
            a = (((-1) ** n) * (x ** (2*n+1))) / factorial(2*n+1)
            result.append(a)

    if func_name == "cos":
        for n in range(n_terms):
            a = (((-1) ** n) * (x ** (2*n))) / factorial(2*n)
            result.append(a)

    result11 = np.sum(result)
    return round(result11, 6)
