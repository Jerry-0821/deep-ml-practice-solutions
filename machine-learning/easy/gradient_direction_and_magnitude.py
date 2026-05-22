"""
Problem:
Gradient Direction and Magnitude.

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

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.

    Args:
        gradient: A list representing the gradient vector

    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    g = np.array(gradient)
    magnitude = np.linalg.norm(g)
    if magnitude == 0:
        zero = [0.0] * len(gradient)
        return {'magnitude': 0.0, 'direction': zero, 'descent_direction': zero}

    direction = (g / magnitude).tolist()
    descent_direction = ( -g / magnitude).tolist()


    return {'magnitude': magnitude, 'direction': direction, 'descent_direction': descent_direction}
