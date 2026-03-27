# GitHub Actions Lab - ML Data Utilities with CI/CD 🧪

## Overview
This lab demonstrates **GitHub Actions** for automated testing (CI) using **pytest** and **unittest** on ML data preprocessing utility functions.

## Based On
Original GitHub Lab-1 from [Prof. Ramin Mohammadi's MLOps Repo](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab1) — Calculator functions with pytest/unittest and GitHub Actions.

## Modifications from Original

| # | Original Lab | This Lab |
|---|----------|----------|
| 1 | `calculator.py` (add, subtract, multiply) | **`data_utils.py`** — ML preprocessing functions |
| 2 | 4 basic arithmetic functions | **5 ML functions**: normalize, standardize, detect_outliers, compute_metrics, train_test_split |
| 3 | 4 simple tests per file | **15 tests** including edge cases and error handling |
| 4 | No error handling tests | **pytest.raises / assertRaises** for invalid inputs |
| 5 | No external dependencies | Uses **NumPy** for statistical computations |
| 6 | Basic assertions only | **Multiple assertion types**: assertEqual, assertAlmostEqual, assertIn, assertRaises |

## How to Run Locally

### Install dependencies
```bash
cd github_actions_lab
pip install -r requirements.txt
```

### Run pytest
```bash
pytest test/test_pytest.py -v
```

### Run unittest
```bash
python -m pytest test/test_unittest.py -v
```

## GitHub Actions (CI/CD)
Two workflows run automatically on every push to `main`:

- **`pytest_action.yml`** — Runs all pytest tests
- **`unittest_action.yml`** — Runs all unittest tests

Check the **Actions** tab in the GitHub repo to see the green checkmarks.

## Functions Tested

| Function | Description |
|----------|-------------|
| `normalize(data)` | Min-max normalization to [0, 1] |
| `standardize(data)` | Z-score standardization (mean=0, std=1) |
| `detect_outliers(data)` | Z-score based outlier detection |
| `compute_metrics(y_true, y_pred)` | Accuracy and error rate calculation |
| `train_test_split_custom(data, labels)` | Custom train/test split |

## Project Structure
```
github_actions_lab/
├── src/
│   ├── __init__.py
│   └── data_utils.py
├── test/
│   ├── __init__.py
│   ├── test_pytest.py
│   └── test_unittest.py
├── data/
│   └── __init__.py
├── requirements.txt
└── README.md

.github/workflows/
├── pytest_action.yml
└── unittest_action.yml
```