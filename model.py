"""
model.py
--------
Core, reusable functions for the Breast Cancer Diagnosis prediction project.

This module handles:
    * Loading the dataset
    * Exploratory data analysis helpers
    * Preprocessing (train/test split + scaling)
    * Training multiple supervised classification models
    * Evaluating those models with standard classification metrics
    * Producing the visualizations used in the report (confusion matrix,
      ROC curve, feature importance)

Running this file directly (`python src/model.py`) executes the full
pipeline end-to-end and writes all metrics/plots into the `results/`
directory so the numbers in the README and notebook are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    The dataset ships with scikit-learn (no internet download required),
    which keeps this project fully reproducible offline. It contains
    measurements computed from digitized images of fine needle aspirate
    (FNA) biopsies of breast masses.

    Returns
    -------
    pd.DataFrame
        Feature columns plus a 'target' column (0 = malignant, 1 = benign)
        and a human-readable 'diagnosis' column.
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    # raw.frame already has a 'target' column; add a readable label too
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    return df


# --------------------------------------------------------------------------- #
# 2. Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split the dataframe into train/test sets and scale numerical features.

    Scaling is fit ONLY on the training data and then applied to the test
    data, to avoid data leakage.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler
    """
    feature_names = [c for c in df.columns if c not in ("target", "diagnosis")]
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


# --------------------------------------------------------------------------- #
# 3. Model training
# --------------------------------------------------------------------------- #
def get_models() -> dict:
    """Return the dictionary of models used for comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE
        ),
    }


def train_models(models: dict, X_train, y_train) -> dict:
    """Fit every model in `models` on the training data."""
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


# --------------------------------------------------------------------------- #
# 4. Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    y_pred: np.ndarray = field(repr=False)
    y_proba: np.ndarray = field(repr=False)


def evaluate_model(name: str, model, X_test, y_test) -> EvalResult:
    """Compute standard classification metrics for a single fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return EvalResult(
        name=name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
        y_pred=y_pred,
        y_proba=y_proba,
    )


def evaluate_all(models: dict, X_test, y_test) -> dict:
    return {name: evaluate_model(name, m, X_test, y_test) for name, m in models.items()}


def results_to_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        rows.append(
            {
                "Model": r.name,
                "Accuracy": round(r.accuracy, 4),
                "Precision": round(r.precision, 4),
                "Recall": round(r.recall, 4),
                "F1-Score": round(r.f1, 4),
                "ROC-AUC": round(r.roc_auc, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Plotting
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(result: EvalResult, y_test, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = confusion_matrix(y_test, result.y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {result.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(results: dict, y_test, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r.y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {r.roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, out_path: Path, top_n: int = 15):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x=importances.values, y=importances.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Full pipeline (used by both the notebook and this script directly)
# --------------------------------------------------------------------------- #
def run_pipeline(save_outputs: bool = True) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

    models = get_models()
    train_models(models, X_train, y_train)

    results = evaluate_all(models, X_test, y_test)
    comparison_df = results_to_dataframe(results)

    if save_outputs:
        class_names = ["malignant", "benign"]

        best_name = comparison_df.iloc[0]["Model"]
        best_result = results[best_name]
        plot_confusion_matrix(
            best_result, y_test, class_names, RESULTS_DIR / "confusion_matrix.png"
        )
        plot_roc_curves(results, y_test, RESULTS_DIR / "roc_curve.png")
        plot_feature_importance(
            models["Random Forest"], feature_names, RESULTS_DIR / "feature_importance.png"
        )

        comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(
                {
                    "best_model": best_name,
                    "comparison": comparison_df.to_dict(orient="records"),
                },
                f,
                indent=2,
            )

    return {
        "df": df,
        "models": models,
        "results": results,
        "comparison_df": comparison_df,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    output = run_pipeline()
    print("\n=== Model Comparison ===")
    print(output["comparison_df"].to_string(index=False))
    print(f"\nBest model: {output['comparison_df'].iloc[0]['Model']}")
    print(f"Results saved to: {RESULTS_DIR}")
