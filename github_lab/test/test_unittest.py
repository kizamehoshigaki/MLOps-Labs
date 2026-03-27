"""
Unittest tests for data_utils.py
Modified from original test_unittest.py (calculator tests)
Changes: Tests for ML preprocessing functions instead of arithmetic
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_utils import normalize, standardize, detect_outliers, compute_metrics, train_test_split_custom


class TestNormalize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(normalize([1, 2, 3, 4, 5]), [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_same_values(self):
        self.assertEqual(normalize([5, 5, 5]), [0.0, 0.0, 0.0])

    def test_negative(self):
        self.assertEqual(normalize([-10, 0, 10]), [0.0, 0.5, 1.0])


class TestStandardize(unittest.TestCase):
    def test_zero_mean(self):
        result = standardize([10, 20, 30, 40, 50])
        self.assertAlmostEqual(np.mean(result), 0.0, places=10)

    def test_same_values(self):
        self.assertEqual(standardize([7, 7, 7]), [0.0, 0.0, 0.0])


class TestDetectOutliers(unittest.TestCase):
    def test_no_outliers(self):
        self.assertEqual(detect_outliers([1, 2, 3, 4, 5]), [])

    def test_with_outlier(self):
        result = detect_outliers([1, 2, 3, 4, 100])
        self.assertIn(4, result)


class TestComputeMetrics(unittest.TestCase):
    def test_perfect(self):
        result = compute_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(result["accuracy"], 1.0)

    def test_half(self):
        result = compute_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(result["accuracy"], 0.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            compute_metrics([1, 0], [1])


class TestTrainTestSplit(unittest.TestCase):
    def test_sizes(self):
        X_train, X_test, y_train, y_test = train_test_split_custom(
            list(range(10)), list(range(10)), test_ratio=0.2
        )
        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            train_test_split_custom([1, 2, 3], [1, 2], test_ratio=0.2)


if __name__ == '__main__':
    unittest.main()