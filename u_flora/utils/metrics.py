def compute_jain_fairness_index(counts: list[int]) -> float:
    """Jain's Fairness Index over a participation count distribution.

    JFI = (sum(x))^2 / (N * sum(x^2))

    Returns 1.0 for perfect fairness (all counts equal) and approaches 0.0 for
    increasing unfairness.
    Returns 1.0 if all counts are zero (undefined, treat as perfectly fair).
    """
    n = len(counts)
    if n == 0:
        return 1.0
    summed = sum(counts)
    if summed == 0:
        return 1.0
    square_summed = sum(x * x for x in counts)
    if square_summed == 0:
        return 1.0
    return (summed * summed) / (n * square_summed)


def compute_gini_coefficient(counts: list[int]) -> float:
    """Gini coefficient of a participation count distribution.

    Returns 0.0 for perfect equality and 1.0 for maximum inequality.
    Returns 0.0 if all counts are zero (undefined, treat as equal).
    """
    n = len(counts)
    if n == 0 or sum(counts) == 0:
        return 0.0
    sorted_counts = sorted(counts)
    cum = 0.0
    for i, x in enumerate(sorted_counts):
        cum += (2 * (i + 1) - n - 1) * x
    return cum / (n * sum(sorted_counts))
