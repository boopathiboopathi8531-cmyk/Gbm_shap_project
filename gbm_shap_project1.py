""" GBM Hyperparameter Tuning + SHAP Explanation Project Filename: gbm_shap_project.py

Description:

Uses sklearn's breast_cancer dataset (binary classification)

Trains a baseline XGBoost classifier

Performs Bayesian hyperparameter tuning using Optuna

Trains final optimized model

Evaluates using ROC AUC and classification report

Generates SHAP summary and force plots (saved as PNGs)


Requirements: pip install numpy pandas scikit-learn xgboost optuna shap matplotlib joblib

Usage: python gbm_shap_project.py

Outputs created in ./outputs/ :

baseline_report.txt

optimized_report.txt

shap_summary.png

shap_feature_importance.png

shap_force_sample0.png

best_params.json

final_model.joblib


"""

import os
import json
import joblib
import optuna
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

#-----------------------------
# Utilities
#-----------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

#-----------------------------
# Data loading and preprocessing
#-----------------------------

def load_and_prepare_data(test_size=0.2, random_state=42, scale=True):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Optional scaling
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, y_train, y_test, scaler

#-----------------------------
# Baseline model
#-----------------------------

def train_baseline(X_train, y_train, X_test, y_test):
    print("Training baseline XGBoost classifier (default hyperparameters)...")
    clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    clf.fit(X_train, y_train)

    preds_proba = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = roc_auc_score(y_test, preds_proba)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    baseline_stats = {
        "auc": float(auc),
        "accuracy": float(acc),
    }

    with open(os.path.join(OUTPUT_DIR, "baseline_report.txt"), "w") as f:
        f.write("Baseline XGBoost Report\n")
        f.write(json.dumps(baseline_stats, indent=2))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print("Baseline AUC: {:.4f}, Accuracy: {:.4f}".format(auc, acc))
    return clf, baseline_stats

#-----------------------------
# Hyperparameter tuning with Optuna
#-----------------------------

def objective(trial, X_train, y_train, X_valid, y_valid):
    # Suggested hyperparameters
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.5),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        # Keep tree_method unset so it works across environments; users can set gpu_hist locally
    }

    clf = XGBClassifier(**param)

    # Removed early_stopping_rounds and verbose from XGBClassifier.fit
    # as it's not directly supported in newer versions for the scikit-learn API.
    # n_estimators will control the number of boosting rounds.
    clf.fit(
        X_train,
        y_train
    )

    preds_proba = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds_proba)

    # We want to maximize AUC
    return auc

def run_optuna_search(X_train, y_train, n_trials=40):
    # Create a small validation split from training data for tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(trial, X_tr, y_tr, X_val, y_val)
    study.optimize(func, n_trials=n_trials, show_progress_bar=True)

    print("Best AUC in tuning: {:.5f}".format(study.best_value))
    print("Best params:\n", study.best_params)

    save_json(study.best_params, os.path.join(OUTPUT_DIR, "best_params.json"))
    return study.best_params

#-----------------------------
# Train final optimized model
#-----------------------------

def train_final_model(best_params, X_train, y_train, X_test, y_test):
    params = best_params.copy()
    # Ensure meta params are set correctly
    params.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42})

    print("Training final model with best params...")
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)

    preds_proba = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = roc_auc_score(y_test, preds_proba)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    optimized_stats = {"auc": float(auc), "accuracy": float(acc)}

    with open(os.path.join(OUTPUT_DIR, "optimized_report.txt"), "w") as f:
        f.write("Optimized XGBoost Report\n")
        f.write(json.dumps(optimized_stats, indent=2))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Save model
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "final_model.joblib"))
    print("Optimized AUC: {:.4f}, Accuracy: {:.4f}".format(auc, acc))
    return clf, optimized_stats

#-----------------------------
# SHAP explanation
#-----------------------------

def shap_analysis(model, X_train, X_test):
    print("Running SHAP analysis (this may take a little while)...")

    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
    plt.title("SHAP feature importance (mean abs SHAP)")
    plt.tight_layout()
    out_path1 = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
    plt.savefig(out_path1, dpi=150)
    plt.close()

    # Summary plot (dot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP summary plot")
    plt.tight_layout()
    out_path2 = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(out_path2, dpi=150)
    plt.close()

    # Force plot for first sample (save as PNG)
    # Force plots are JavaScript/HTML in many versions; convert to matplotlib via shap.plots._force_matplotlib if available
    try:
        # shap.force_plot returns an object that can be saved to HTML; to save PNG we can use matplotlib wrapper when available
        force_fig = shap.plots.force(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], matplotlib=True)
        # Save the matplotlib figure
        plt.tight_layout()
        out_path3 = os.path.join(OUTPUT_DIR, "shap_force_sample0.png")
        plt.savefig(out_path3, dpi=200)
        plt.close()
    except Exception as e:
        # Fallback: save HTML representation
        try:
            html_path = os.path.join(OUTPUT_DIR, "shap_force_sample0.html")
            shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], matplotlib=False, show=False)
            # If shap returns an object, we can save via shap.save_html in newer versions — but not all versions have it
            print("Could not save matplotlib force plot (will skip PNG). Error:", str(e))
        except Exception:
            print("SHAP force plot unavailable in this environment; consider opening an interactive notebook to view force plots.")

    return {
        "feature_importance_png": out_path1,
        "summary_png": out_path2,
        # "force_png": out_path3  # may not exist depending on SHAP version
    }

#-----------------------------
# Main pipeline
#-----------------------------

def main():
    # 1. Load and prepare data
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()

    # 2. Baseline model
    baseline_model, baseline_stats = train_baseline(X_train, y_train, X_test, y_test)

    # 3. Hyperparameter tuning
    best_params = run_optuna_search(X_train, y_train, n_trials=40)

    # 4. Train final model
    final_model, optimized_stats = train_final_model(best_params, X_train, y_train, X_test, y_test)

    # 5. SHAP analysis
    shap_outputs = shap_analysis(final_model, X_train, X_test)

    # 6. Save summary
    summary = {
        "baseline": baseline_stats,
        "optimized": optimized_stats,
        "shap_outputs": shap_outputs,
    }
    save_json(summary, os.path.join(OUTPUT_DIR, "summary_stats.json"))

    print("\nAll done. Outputs saved to: {}".format(os.path.abspath(OUTPUT_DIR)))

if __name__ == "__main__":
    main()
