"""Classification model training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from ml_portfolio.shared.utils import RANDOM_STATE


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    """Train a balanced logistic regression model."""
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    """Train a balanced random forest."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> XGBClassifier:
    """Train an XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    model_name: str = "Model",
) -> dict[str, float]:
    """Evaluate a classification model.

    Returns accuracy, precision, recall, f1, balanced_accuracy, and roc_auc.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }

    print(f"\n{model_name}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:20s}: {v:.4f}")

    return metrics


def save_metrics(metrics: list[dict[str, Any]], path: Path) -> None:
    """Save evaluation metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_model(model: Any, path: Path) -> None:
    """Persist a trained model with joblib."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
