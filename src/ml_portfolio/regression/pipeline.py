"""End-to-end regression pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ml_portfolio.regression.features import (
    apply_one_hot_encoding,
    apply_ordinal_encoding,
    create_composite_features,
    drop_original_columns,
    extract_job_features,
    extract_timestamp_features,
    scale_features,
)
from ml_portfolio.regression.models import (
    evaluate_model,
    save_metrics,
    save_model,
    train_ridge,
    train_stacking,
    train_xgboost,
)
from ml_portfolio.regression.preprocessing import (
    apply_log_transform,
    clean_data,
    load_raw_data,
    split_data,
)
from ml_portfolio.shared.utils import RANDOM_STATE, Timer, get_default_paths


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Run full feature engineering on a Polars DataFrame."""
    df = apply_ordinal_encoding(df)
    df = apply_one_hot_encoding(df)
    df = extract_job_features(df)
    df = extract_timestamp_features(df)
    df = create_composite_features(df)
    df = drop_original_columns(df)
    return df


def run_pipeline(
    raw_path: Path | None = None,
    cleaned_dir: Path | None = None,
    models_dir: Path | None = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run the complete regression pipeline.

    Args:
        raw_path: Path to raw salary-survey.csv. Defaults to repo canonical path.
        cleaned_dir: Directory to save cleaned splits. Defaults to repo canonical path.
        models_dir: Directory to save trained models. Defaults to repo canonical path.
        random_state: Reproducibility seed.

    Returns:
        Dict with metrics and model paths.
    """
    paths = get_default_paths("regression")
    raw_path = raw_path or paths["raw_dir"] / "salary-survey.csv"
    cleaned_dir = cleaned_dir or paths["cleaned_dir"]
    models_dir = models_dir or paths["models_dir"]

    # ------------------------------------------------------------------
    # 1. Load & clean
    # ------------------------------------------------------------------
    with Timer("Load & Clean"):
        df = load_raw_data(raw_path)
        df = clean_data(df)
        df = apply_log_transform(df)
        train_df, val_df, test_df = split_data(df, random_state=random_state)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    with Timer("Feature Engineering"):
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        target_col = "log_salary"
        y_train = train_df[target_col].to_numpy()
        y_val = val_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()

        X_train = train_df.drop(target_col)
        X_val = val_df.drop(target_col)
        X_test = test_df.drop(target_col)

        assert (
            X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        ), "Feature mismatch across splits"

        X_train_s, X_val_s, X_test_s, scaler = scale_features(
            X_train,
            X_val,
            X_test,
            scaler_path=models_dir / "scaler.pkl",
            metadata_path=models_dir / "scaler_metadata.json",
        )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    with Timer("Train Models"):
        X_train_np = X_train_s.to_numpy()
        X_val_np = X_val_s.to_numpy()
        X_test_np = X_test_s.to_numpy()

        ridge = train_ridge(X_train_np, y_train)
        xgb = train_xgboost(X_train_np, y_train)
        stacking = train_stacking(X_train_np, y_train)

        models = {
            "ridge": ridge,
            "xgboost": xgb,
            "stacking": stacking,
        }

        for name, model in models.items():
            save_model(model, models_dir / f"{name}_model.pkl")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    with Timer("Evaluate"):
        all_metrics = []
        for name, model in models.items():
            metrics = evaluate_model(model, X_test_np, y_test, model_name=name)
            all_metrics.append(metrics)

        save_metrics(all_metrics, models_dir / "metrics.json")

    return {
        "models": {k: str(models_dir / f"{k}_model.pkl") for k in models},
        "metrics": all_metrics,
        "scaler": str(models_dir / "scaler.pkl"),
    }


if __name__ == "__main__":
    run_pipeline()
