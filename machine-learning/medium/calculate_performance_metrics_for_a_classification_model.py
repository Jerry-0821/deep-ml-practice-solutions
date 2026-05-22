"""
Problem:
Calculate Performance Metrics for a Classification Model.

Topic:
Classification metrics.

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

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:


    TP = sum(a==1 and p==1 for a, p in zip(actual, predicted))
    TN = sum(a==0 and p==0 for a, p in zip(actual, predicted))
    FP = sum(a==0 and p==1 for a, p in zip(actual, predicted))
    FN = sum(a==1 and p==0 for a, p in zip(actual, predicted))

    confusion_matrix = [[TP, FN], [FP, TN]]
    accuracy    = (TP+TN) / (TP+TN+FP+FN) if (TP+TN+FP+FN) else 0.0
    precision   = TP / (TP+FP) if (TP+FP) else 0.0
    recall      = TP / (TP+FN) if (TP+FN) else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = TN / (TN+FP) if (TN+FP) else 0.0
    npv         = TN / (TN+FN) if (TN+FN) else 0.0

    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(npv, 3)
