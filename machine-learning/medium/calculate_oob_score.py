"""
Problem:
Calculate OOB Score.

Topic:
Ensemble methods.

Idea:
For each estimator, collect predictions for samples that were not included in
that estimator's bootstrap sample, then score majority-vote predictions.

Key Steps:
1. Build an out-of-bag prediction list for each sample.
2. Add predictions only from estimators where the sample was not in-bag.
3. Majority vote each sample's OOB predictions and compute accuracy.

Complexity:
Time: O(e * n), where e is estimators and n is samples.
Space: O(e * n) in the worst case for stored OOB predictions.
"""

from collections import Counter


def calculate_oob_score(n_samples: int, bootstrap_indices: list, predictions: list, y_true: list) -> float:
    """
    Calculate the out-of-bag accuracy score for a bagging ensemble.
    """
    oob_preds = [[] for _ in range(n_samples)]
    for estimator_idx, in_bag in enumerate(bootstrap_indices):
        in_bag_set = set(in_bag)
        for sample_idx in range(n_samples):
            # A sample is out-of-bag for this estimator if it is not in the bootstrap set.
            if sample_idx not in in_bag_set:
                oob_preds[sample_idx].append(predictions[estimator_idx][sample_idx])

    correct, total = 0, 0
    for sample_idx in range(n_samples):
        if not oob_preds[sample_idx]:
            continue
        majority_vote = Counter(oob_preds[sample_idx]).most_common(1)[0][0]
        correct += majority_vote == y_true[sample_idx]
        total += 1

    return correct / total if total > 0 else 0.0
