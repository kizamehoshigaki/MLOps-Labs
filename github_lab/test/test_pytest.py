"""
Pytest tests for data_utils.py
Modified from original test_pytest.py (calculator tests)
Changes: Tests for ML preprocessing functions instead of arithmetic
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_utils import normalize, standardize, detect_outliers, compute_metrics, train_test_split_custom


# --- Test normalize ---
def test_normalize_basic():
    result = normalize([1, 2, 3, 4, 5])
    assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

def test_normalize_same_values():
    result = normalize([5, 5, 5])
    assert result == [0.0, 0.0, 0.0]

def test_normalize_negative():
    result = normalize([-10, 0, 10])
    assert result == [0.0, 0.5, 1.0]


# --- Test standardize ---
def test_standardize_output_mean():
    import numpy as np
    result = standardize([10, 20, 30, 40, 50])
    assert abs(np.mean(result)) < 1e-10

def test_standardize_same_values():
    result = standardize([7, 7, 7])
    assert result == [0.0, 0.0, 0.0]


# --- Test detect_outliers ---
def test_detect_outliers_no_outliers():
    result = detect_outliers([1, 2, 3, 4, 5])
    assert result == []

def test_detect_outliers_with_outlier():
    result = detect_outliers([1, 2, 3, 4, 100], threshold=1.5)
    assert 4 in result  # index 4 (value 100) is an outlier

def test_detect_outliers_same_values():
    result = detect_outliers([5, 5, 5, 5])
    assert result == []


# --- Test compute_metrics ---
def test_compute_metrics_perfect():
    result = compute_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert result["accuracy"] == 1.0
    assert result["error_rate"] == 0.0

def test_compute_metrics_half():
    result = compute_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["accuracy"] == 0.5

def test_compute_metrics_length_mismatch():
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [1])

def test_compute_metrics_empty():
    with pytest.raises(ValueError):
        compute_metrics([], [])


# --- Test train_test_split_custom ---
def test_split_sizes():
    X_train, X_test, y_train, y_test = train_test_split_custom(
        list(range(10)), list(range(10)), test_ratio=0.2
    )
    assert len(X_train) == 8
    assert len(X_test) == 2

def test_split_length_mismatch():
    with pytest.raises(ValueError):
        train_test_split_custom([1, 2, 3], [1, 2], test_ratio=0.2)