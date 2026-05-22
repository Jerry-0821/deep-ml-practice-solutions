"""
Problem:
Linear Regression - Power Grid Optimization.

Topic:
Regression.

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

import math

PI = 3.14159


def power_grid_forecast(consumption_data):
    # 1) Subtract the daily fluctuation (10 * sin(2 * PI * day / 10)) from each data point.
    # 2) Perform linear regression on the detrended data.
    # 3) Predict day 15's base consumption.
    # 4) Add the day 15 fluctuation back.
    # 5) Round, then add a 5% safety margin (rounded up).
    # 6) Return the final integer.

    n = len(consumption_data)

    x = []
    y_clean = []

    for i in range(n):
        day = i + 1
        f = 10 * math.sin(2 * PI * day / 10)
        detrended = consumption_data[i] - f
        x.append(day)
        y_clean.append(detrended)

    x_mean = sum(x) / n
    y_mean = sum(y_clean) / n

    numerator = sum((x[i] - x_mean) * (y_clean[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

    m = numerator / denominator
    b = y_mean - (m * x_mean)

    base15 = m * 15 + b

    f15 = 10 * math.sin(2 * PI * 15 / 10)
    pred15 = base15 + f15
    final15 = math.ceil(1.05 * pred15)

    return final15
