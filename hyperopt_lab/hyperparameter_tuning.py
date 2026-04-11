"""
Hyperparameter Tuning Lab - Modified Version
Original: HyperOpt lab from Prof. Ramin Mohammadi's MLOps repo
Modifications:
    1. Different dataset: Diabetes dataset instead of original
    2. Different tuning library: Optuna instead of HyperOpt
    3. Multiple models: XGBoost, Random Forest, Gradient Boosting
    4. Added visualization: optimization history, parameter importance
    5. Added MLflow integration for logging best runs
    6. Comparison of tuned vs untuned models
"""

import optuna
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Step 1: Load and Prepare Data
# ============================================================
logger.info("Loading Diabetes dataset...")
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logger.info(f"Train: {X_train_scaled.shape[0]} samples, Test: {X_test_scaled.shape[0]} samples")

# ============================================================
# Step 2: Baseline Models (No Tuning)
# ============================================================
logger.info("\n=== Baseline Models (Default Params) ===")

baselines = {}
for name, model in [
    ("RandomForest", RandomForestRegressor(random_state=42)),
    ("GradientBoosting", GradientBoostingRegressor(random_state=42))
]:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    baselines[name] = {"MSE": mse, "MAE": mae, "R2": r2}
    logger.info(f"  {name}: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

# ============================================================
# Step 3: Optuna Hyperparameter Tuning - Random Forest
# ============================================================
logger.info("\n=== Tuning Random Forest with Optuna (50 trials) ===")

def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
    }
    model = RandomForestRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()

rf_study = optuna.create_study(direction='minimize', study_name='random_forest')
rf_study.optimize(rf_objective, n_trials=50)

logger.info(f"  Best MSE (CV): {rf_study.best_value:.2f}")
logger.info(f"  Best Params: {rf_study.best_params}")

# ============================================================
# Step 4: Optuna Hyperparameter Tuning - Gradient Boosting
# ============================================================
logger.info("\n=== Tuning Gradient Boosting with Optuna (50 trials) ===")

def gb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()

gb_study = optuna.create_study(direction='minimize', study_name='gradient_boosting')
gb_study.optimize(gb_objective, n_trials=50)

logger.info(f"  Best MSE (CV): {gb_study.best_value:.2f}")
logger.info(f"  Best Params: {gb_study.best_params}")

# ============================================================
# Step 5: Evaluate Tuned Models on Test Set
# ============================================================
logger.info("\n=== Evaluating Tuned Models on Test Set ===")

tuned_results = {}

# Tuned Random Forest
rf_tuned = RandomForestRegressor(**rf_study.best_params, random_state=42)
rf_tuned.fit(X_train_scaled, y_train)
y_pred_rf = rf_tuned.predict(X_test_scaled)
tuned_results["RandomForest_Tuned"] = {
    "MSE": mean_squared_error(y_test, y_pred_rf),
    "MAE": mean_absolute_error(y_test, y_pred_rf),
    "R2": r2_score(y_test, y_pred_rf)
}

# Tuned Gradient Boosting
gb_tuned = GradientBoostingRegressor(**gb_study.best_params, random_state=42)
gb_tuned.fit(X_train_scaled, y_train)
y_pred_gb = gb_tuned.predict(X_test_scaled)
tuned_results["GradientBoosting_Tuned"] = {
    "MSE": mean_squared_error(y_test, y_pred_gb),
    "MAE": mean_absolute_error(y_test, y_pred_gb),
    "R2": r2_score(y_test, y_pred_gb)
}

# ============================================================
# Step 6: Log Best Models to MLflow
# ============================================================
logger.info("\n=== Logging to MLflow ===")
mlflow.set_experiment("hyperparameter_tuning")

for name, study, model in [
    ("RF_Tuned", rf_study, rf_tuned),
    ("GB_Tuned", gb_study, gb_tuned)
]:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(study.best_params)
        metrics = tuned_results[f"{'RandomForest' if 'RF' in name else 'GradientBoosting'}_Tuned"]
        mlflow.log_metric("mse", metrics["MSE"])
        mlflow.log_metric("mae", metrics["MAE"])
        mlflow.log_metric("r2", metrics["R2"])
        mlflow.sklearn.log_model(model, "model")

# ============================================================
# Step 7: Visualization
# ============================================================
logger.info("\n=== Generating Plots ===")

# Optimization history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, study, name in [(axes[0], rf_study, "Random Forest"), (axes[1], gb_study, "Gradient Boosting")]:
    trials = [t.value for t in study.trials]
    ax.plot(trials, marker='.', alpha=0.6)
    ax.set_xlabel("Trial")
    ax.set_ylabel("MSE (CV)")
    ax.set_title(f"Optimization History - {name}")
    best_so_far = [min(trials[:i+1]) for i in range(len(trials))]
    ax.plot(best_so_far, color='red', linewidth=2, label='Best so far')
    ax.legend()
plt.tight_layout()
plt.savefig("optimization_history.png", dpi=100, bbox_inches='tight')
plt.close()
logger.info("  Saved optimization_history.png")

# Baseline vs Tuned comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = ['RF Baseline', 'RF Tuned', 'GB Baseline', 'GB Tuned']
mse_vals = [
    baselines["RandomForest"]["MSE"], tuned_results["RandomForest_Tuned"]["MSE"],
    baselines["GradientBoosting"]["MSE"], tuned_results["GradientBoosting_Tuned"]["MSE"]
]
colors = ['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71']
bars = ax.bar(models, mse_vals, color=colors)
ax.set_ylabel("Mean Squared Error")
ax.set_title("Baseline vs Tuned Model Comparison")
for bar, val in zip(bars, mse_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val:.1f}',
            ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
logger.info("  Saved model_comparison.png")

# ============================================================
# Step 8: Print Summary
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
all_results = {**{f"{k}_Baseline": v for k, v in baselines.items()}, **tuned_results}
df_results = pd.DataFrame(all_results).T
df_results.index.name = "Model"
print(df_results.to_string())
print("=" * 70)
print("\nPlots saved: optimization_history.png, model_comparison.png")
print("MLflow logs saved. Run 'mlflow ui' to view.")