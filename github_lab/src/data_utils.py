"""
Data Utilities for ML Preprocessing
Modified from original calculator.py in GitHub Lab-1
Changes: ML-focused functions instead of basic arithmetic
"""

import numpy as np


def normalize(data):
    """Min-max normalize a list of numbers to [0, 1] range."""
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.0] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def standardize(data):
    """Standardize data to zero mean and unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return [0.0] * len(data)
    return [(x - mean) / std for x in data]


def detect_outliers(data, threshold=2.0):
    """Detect outliers using z-score method. Returns indices of outliers."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return []
    outliers = []
    for i, x in enumerate(data):
        z_score = abs((x - mean) / std)
        if z_score > threshold:
            outliers.append(i)
    return outliers


def compute_metrics(y_true, y_pred):
    """Compute accuracy and error rate for classification predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Input lists cannot be empty")
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / len(y_true)
    error_rate = 1 - accuracy
    return {"accuracy": round(accuracy, 4), "error_rate": round(error_rate, 4)}


def train_test_split_custom(data, labels, test_ratio=0.2):
    """Split data into train and test sets."""
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")
    split_idx = int(len(data) * (1 - test_ratio))
    X_train = data[:split_idx]
    X_test = data[split_idx:]
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]
    return X_train, X_test, y_train, y_test