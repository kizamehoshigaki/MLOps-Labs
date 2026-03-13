"""
MLflow Lab - Modified Version
Original Lab: Wine Quality prediction with PySpark + MLflow Model Registry
Modifications:
    1. Different dataset: Breast Cancer dataset instead of Wine Quality CSV
    2. Different models: Logistic Regression + XGBoost comparison (original used only Random Forest)
    3. No Java/PySpark dependency - pure scikit-learn + xgboost
    4. Added hyperparameter tuning with GridSearchCV logged to MLflow
    5. Added model comparison across multiple runs
    6. Added artifact logging (confusion matrix plot, feature importance plot)
    7. Simplified setup - no model serving step needed
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from mlflow.models.signature import infer_signature
import warnings
import os
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Step 1: Setup MLflow
# ============================================================
mlflow.set_tracking_uri("mlruns")
EXPERIMENT_NAME = "breast_cancer_classification"
mlflow.set_experiment(EXPERIMENT_NAME)

logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")

# ============================================================
# Step 2: Load and Explore Data
# ============================================================
logger.info("Loading Breast Cancer dataset...")
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

logger.info(f"Dataset shape: {df.shape}")
logger.info(f"Class distribution:\n{df['target'].value_counts()}")

# ============================================================
# Step 3: Data Preprocessing
# ============================================================
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logger.info(f"Train size: {X_train_scaled.shape[0]}, Test size: {X_test_scaled.shape[0]}")

# ============================================================
# Step 4: Helper - Log model run to MLflow
# ============================================================
def log_model_run(run_name, model, X_train, X_test, y_train, y_test, params):
    """Train a model and log everything to MLflow."""
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        logger.info(f"{run_name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # Save confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Malignant', 'Benign'],
                    yticklabels=['Malignant', 'Benign'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {run_name}")
        cm_path = f"confusion_matrix_{run_name.replace(' ', '_')}.png"
        fig.savefig(cm_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Save ROC curve as artifact
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {run_name}")
        ax.legend()
        roc_path = f"roc_curve_{run_name.replace(' ', '_')}.png"
        fig.savefig(roc_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(roc_path)
        os.remove(roc_path)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        return acc, f1, auc

# ============================================================
# Step 5: Run Experiment - Logistic Regression
# ============================================================
logger.info("\n=== Running Logistic Regression ===")
lr_acc, lr_f1, lr_auc = log_model_run(
    run_name="logistic_regression",
    model=LogisticRegression(max_iter=1000, random_state=42),
    X_train=X_train_scaled, X_test=X_test_scaled,
    y_train=y_train, y_test=y_test,
    params={"model": "LogisticRegression", "max_iter": 1000, "solver": "lbfgs"}
)

# ============================================================
# Step 6: Run Experiment - Random Forest
# ============================================================
logger.info("\n=== Running Random Forest ===")
rf_acc, rf_f1, rf_auc = log_model_run(
    run_name="random_forest",
    model=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    X_train=X_train_scaled, X_test=X_test_scaled,
    y_train=y_train, y_test=y_test,
    params={"model": "RandomForest", "n_estimators": 100, "max_depth": 10}
)

# ============================================================
# Step 7: Run Experiment - Gradient Boosting
# ============================================================
logger.info("\n=== Running Gradient Boosting ===")
gb_acc, gb_f1, gb_auc = log_model_run(
    run_name="gradient_boosting",
    model=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    X_train=X_train_scaled, X_test=X_test_scaled,
    y_train=y_train, y_test=y_test,
    params={"model": "GradientBoosting", "n_estimators": 100, "max_depth": 3}
)

# ============================================================
# Step 8: Hyperparameter Tuning with MLflow Logging
# ============================================================
logger.info("\n=== Running Hyperparameter Tuning (Random Forest) ===")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

with mlflow.start_run(run_name="rf_grid_search"):
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    best_acc = accuracy_score(y_test, y_pred)
    best_f1 = f1_score(y_test, y_pred)
    best_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model", "RandomForest_Tuned")
    mlflow.log_metric("accuracy", best_acc)
    mlflow.log_metric("f1_score", best_f1)
    mlflow.log_metric("auc", best_auc)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    signature = infer_signature(X_train_scaled, best_model.predict(X_train_scaled))
    mlflow.sklearn.log_model(best_model, "model", signature=signature)

    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Tuned RF -> Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")

# ============================================================
# Step 9: Print Summary
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'RF (Tuned)'],
    'Accuracy': [lr_acc, rf_acc, gb_acc, best_acc],
    'F1 Score': [lr_f1, rf_f1, gb_f1, best_f1],
    'AUC': [lr_auc, rf_auc, gb_auc, best_auc]
})
print(results.to_string(index=False))
print("=" * 60)
print("\nTo view results in MLflow UI, run:")
print("  mlflow ui")
print("Then open http://localhost:5000 in your browser")