"""Regression model training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from ml_portfolio.shared.utils import RANDOM_STATE


def inverse_transform_salary(log_salary: np.ndarray) -> np.ndarray:
    """Invert log1p transform to get original USD salary."""
    return np.expm1(log_salary)


def train_ridge(
    X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0
) -> Ridge:
    """Train a Ridge regression model."""
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> XGBRegressor:
    """Train an XGBoost regressor."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_stacking(
    X_train: np.ndarray, y_train: np.ndarray
) -> Any:
    """Train a stacking ensemble (Ridge + XGBoost -> Ridge meta-learner)."""
    from sklearn.ensemble import StackingRegressor

    estimators = [
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        (
            "xgb",
            XGBRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.1, random_state=RANDOM_STATE),
        cv=5,
        passthrough=True,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true_log: np.ndarray,
    model_name: str = "Model",
) -> dict[str, float]:
    """Evaluate a regression model on both log and original salary scales.

    Returns a dict with RMSE, MAE, and R2 for both scales.
    """
    y_pred_log = model.predict(X)

    # Log-scale metrics
    mse_log = mean_squared_error(y_true_log, y_pred_log)
    rmse_log = float(np.sqrt(mse_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    # Original-scale metrics
    y_true_orig = inverse_transform_salary(y_true_log)
    y_pred_orig = inverse_transform_salary(y_pred_log)

    mse_orig = mean_squared_error(y_true_orig, y_pred_orig)
    rmse_orig = float(np.sqrt(mse_orig))
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    r2_orig = r2_score(y_true_orig, y_pred_orig)

    metrics = {
        "model": model_name,
        "rmse_log": round(rmse_log, 4),
        "mae_log": round(mae_log, 4),
        "r2_log": round(r2_log, 4),
        "rmse_usd": round(rmse_orig, 2),
        "mae_usd": round(mae_orig, 2),
        "r2_usd": round(r2_orig, 4),
    }

    print(f"\n{model_name}")
    print(f"  Log scale   -> RMSE: {rmse_log:.4f}, MAE: {mae_log:.4f}, R2: {r2_log:.4f}")
    print(f"  USD scale   -> RMSE: ${rmse_orig:,.2f}, MAE: ${mae_orig:,.2f}, R2: {r2_orig:.4f}")

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
