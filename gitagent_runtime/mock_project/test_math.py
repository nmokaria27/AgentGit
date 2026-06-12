import pytest
from utils.math_processor import compute_percentage, compute_ratio, compute_weight


def test_percentage_normal():
    assert compute_percentage(50, 200) == 25.0
    assert compute_percentage(1, 4) == 25.0


def test_percentage_zero():
    """compute_percentage must return 0.0 when total is 0, not raise."""
    assert compute_percentage(50, 0) == 0.0


def test_ratio_normal():
    assert compute_ratio(1, 4) == 0.25
    assert compute_ratio(3, 3) == 1.0


def test_ratio_zero():
    """compute_ratio must return 0.0 when denominator is 0, not raise."""
    assert compute_ratio(5, 0) == 0.0


def test_weight_normal():
    assert compute_weight(3, 12) == 0.25
    assert compute_weight(0, 10) == 0.0


def test_weight_zero():
    """compute_weight must return 0.0 when total_weight is 0, not raise."""
    assert compute_weight(3, 0) == 0.0
