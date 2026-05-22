"""
Problem:
Gaussian Naive Bayes Classifier.

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

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.

    Args:
        X_train: Training features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test features (shape: N_test x D)

    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    classes = np.unique(y_train)
    n = len(y_train)

    stats = {}
    for c in classes:
        X_c = X_train[y_train == c]
        stats[c] = {
            "prior": np.log(len(X_c) / n),
            "mean": X_c.mean(axis=0),
            "var": X_c.var(axis=0) + 1e-9
        }

    preds = []
    for x in X_test:
        log_posteriors = []
        for c in classes:
            m = stats[c]["mean"]
            v = stats[c]["var"]
            prior = stats[c]["prior"]
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * v) + (x - m)**2 / v)
            log_posteriors.append(prior + log_likelihood)

        preds.append(classes[np.argmax(log_posteriors)])
    return np.array(preds)
