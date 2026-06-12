def compute_percentage(part, total):
    """Returns what percentage `part` is of `total`."""
    if total == 0:
        return 0.0
    return (part / total) * 100


def compute_ratio(numerator, denominator):
    """Returns decimal ratio of numerator to denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_weight(value, total_weight):
    """Returns fractional weight of value within total_weight."""
    if total_weight == 0:
        return 0.0
    return value / total_weight
