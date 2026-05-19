"""End-to-end classification pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ml_portfolio.classification.features import (
    aggregate_features,
    derive_interaction_features,
    derive_temporal_features,
    derive_text_features,
    encode_categoricals,
    scale_features,
)
from ml_portfolio.classification.models import (
    evaluate_model,
    save_metrics,
    save_model,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from ml_portfolio.classification.preprocessing import (
    TARGET_COL,
    balance_train_data,
    clean_data,
    load_raw_data,
    split_data,
)
from ml_portfolio.shared.utils import RANDOM_STATE, Timer, get_default_paths


def engineer_features(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run full feature engineering on all three splits."""
    for fn in [
        derive_text_features,
        derive_temporal_features,
        derive_interaction_features,
        aggregate_features,
    ]:
        df_train = fn(df_train)
        df_val = fn(df_val)
        df_test = fn(df_test)

    return df_train, df_val, df_test


def run_pipeline(
    raw_dir: Path | None = None,
    cleaned_dir: Path | None = None,
    models_dir: Path | None = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run the complete classification pipeline.

    Args:
        raw_dir: Directory containing airbnb_timepoint1.csv … 4.csv.
        cleaned_dir: Directory to save cleaned splits.
        models_dir: Directory to save trained models.
        random_state: Reproducibility seed.

    Returns:
        Dict with metrics and model paths.
    """
    paths = get_default_paths("classification")
    raw_dir = raw_dir or paths["raw_dir"]
    cleaned_dir = cleaned_dir or paths["cleaned_dir"]
    models_dir = models_dir or paths["models_dir"]

    # ------------------------------------------------------------------
    # 1. Load & clean
    # ------------------------------------------------------------------
    with Timer("Load & Clean"):
        df = load_raw_data(raw_dir)
        df = clean_data(df)
        train_df, val_df, test_df = split_data(df, random_state=random_state)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    with Timer("Feature Engineering"):
        train_df, val_df, test_df = engineer_features(train_df, val_df, test_df)

        y_train = train_df[TARGET_COL].to_numpy()
        y_val = val_df[TARGET_COL].to_numpy()
        y_test = test_df[TARGET_COL].to_numpy()

        X_train = train_df.drop(TARGET_COL)
        X_val = val_df.drop(TARGET_COL)
        X_test = test_df.drop(TARGET_COL)

        # Identify categorical columns (still object/string dtype or already encoded)
        # At this point, categoricals are still string; we need to encode them.
        categorical_cols = [
            c for c in X_train.columns if X_train[c].dtype == pl.Utf8
        ]
        categorical_indices = [X_train.columns.index(c) for c in categorical_cols]

        X_train, X_val, X_test, encoders = encode_categoricals(
            X_train,
            X_val,
            X_test,
            encoders_path=models_dir / "encoders.json",
        )

        # Re-identify categorical indices after encoding (should be none left)
        # But some encoded cols may be Int*; we track original indices for SMOTENC.
        # Since we already encoded, SMOTENC needs the original indices.
        # We'll save metadata and proceed with numerical scaling.

        X_train_s, X_val_s, X_test_s, scaler = scale_features(
            X_train,
            X_val,
            X_test,
            scaler_path=models_dir / "scaler.pkl",
        )

    # ------------------------------------------------------------------
    # 3. Balance training data
    # ------------------------------------------------------------------
    with Timer("Balance Training Data"):
        X_train_np = X_train_s.to_numpy()
        X_val_np = X_val_s.to_numpy()
        X_test_np = X_test_s.to_numpy()

        # SMOTENC requires categorical indices on the *original* mixed-type data.
        # Since we already one-hot encoded, we skip SMOTENC here and use a
        # simplified balancing approach for the production pipeline.
        # For full fidelity, run the notebook pipeline.
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline

        sampler = ImbPipeline(
            [
                ("smote", SMOTE(sampling_strategy=0.5, random_state=random_state, k_neighbors=5)),
                ("under", RandomUnderSampler(sampling_strategy=0.75, random_state=random_state)),
            ]
        )
        X_train_bal, y_train_bal = sampler.fit_resample(X_train_np, y_train)
        print(f"Balanced train size: {len(X_train_bal)}")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    with Timer("Train Models"):
        lr = train_logistic_regression(X_train_bal, y_train_bal)
        rf = train_random_forest(X_train_bal, y_train_bal)
        xgb = train_xgboost(X_train_bal, y_train_bal)

        models = {
            "logistic_regression": lr,
            "random_forest": rf,
            "xgboost": xgb,
        }

        for name, model in models.items():
            save_model(model, models_dir / f"{name}.pkl")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    with Timer("Evaluate"):
        all_metrics = []
        for name, model in models.items():
            metrics = evaluate_model(model, X_test_np, y_test, model_name=name)
            all_metrics.append(metrics)

        save_metrics(all_metrics, models_dir / "metrics.json")

    return {
        "models": {k: str(models_dir / f"{k}.pkl") for k in models},
        "metrics": all_metrics,
        "scaler": str(models_dir / "scaler.pkl"),
    }


if __name__ == "__main__":
    run_pipeline()
