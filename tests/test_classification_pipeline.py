"""Smoke tests for the classification pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl

from ml_portfolio.classification.features import (
    derive_temporal_features,
    derive_text_features,
    encode_categoricals,
    scale_features,
)
from ml_portfolio.classification.models import evaluate_model, train_logistic_regression
from ml_portfolio.classification.preprocessing import (
    clean_data,
    load_raw_data,
    split_data,
)
from ml_portfolio.shared.utils import get_default_paths


def test_load_raw_data() -> None:
    paths = get_default_paths("classification")
    df = load_raw_data(paths["raw_dir"])
    assert len(df) > 0
    # Should have data from multiple timepoints
    assert "id" in df.columns
    assert "host_is_superhost" in df.columns


def test_clean_data() -> None:
    paths = get_default_paths("classification")
    df = load_raw_data(paths["raw_dir"])
    df_clean = clean_data(df)
    assert "host_is_superhost" in df_clean.columns
    # No null targets
    assert df_clean["host_is_superhost"].null_count() == 0
    # Boolean target
    assert df_clean["host_is_superhost"].dtype == pl.Boolean


def test_split_data() -> None:
    paths = get_default_paths("classification")
    df = load_raw_data(paths["raw_dir"])
    df_clean = clean_data(df)
    train, val, test = split_data(df_clean)

    total_ids = (
        len(train["id"].unique())
        + len(val["id"].unique())
        + len(test["id"].unique())
    )
    # Should equal unique ids across all splits
    all_ids = len(
        pl.concat([train, val, test])
        .select("id")
        .unique()
    )
    assert total_ids == all_ids

    # Rough 70/15/15
    combined = pl.concat([train, val, test])
    assert len(combined) == len(df_clean)


def test_feature_engineering() -> None:
    paths = get_default_paths("classification")
    df = load_raw_data(paths["raw_dir"])
    df_clean = clean_data(df)
    train, val, test = split_data(df_clean)

    train = derive_text_features(train)
    train = derive_temporal_features(train)

    assert "description_length" in train.columns
    assert "host_tenure_days" in train.columns


def test_encode_and_scale() -> None:
    paths = get_default_paths("classification")
    df = load_raw_data(paths["raw_dir"])
    df_clean = clean_data(df)
    train, val, test = split_data(df_clean)

    for fn in [derive_text_features, derive_temporal_features]:
        train = fn(train)
        val = fn(val)
        test = fn(test)

    y_train = train["host_is_superhost"].to_numpy()
    X_train = train.drop("host_is_superhost")
    X_val = val.drop("host_is_superhost")
    X_test = test.drop("host_is_superhost")

    X_train, X_val, X_test, encoders = encode_categoricals(X_train, X_val, X_test)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    assert X_train_s.shape[1] == X_val_s.shape[1] == X_test_s.shape[1]
    assert len(encoders) > 0
    # No remaining string columns
    for col in X_train_s.columns:
        assert X_train_s[col].dtype != pl.Utf8


def test_train_and_evaluate_lr() -> None:
    paths = get_default_paths("classification")
    df = load_raw_data(paths["raw_dir"])
    df_clean = clean_data(df)
    train, val, test = split_data(df_clean)

    for fn in [derive_text_features, derive_temporal_features]:
        train = fn(train)
        test = fn(test)

    y_train = train["host_is_superhost"].to_numpy()
    y_test = test["host_is_superhost"].to_numpy()
    X_train = train.drop("host_is_superhost")
    X_test = test.drop("host_is_superhost")

    X_train, X_test, _, _ = encode_categoricals(X_train, X_test, X_test)
    X_train_s, X_test_s, _, _ = scale_features(X_train, X_test, X_test)

    X_train_np = np.nan_to_num(X_train_s.to_numpy(), nan=0.0)
    X_test_np = np.nan_to_num(X_test_s.to_numpy(), nan=0.0)

    model = train_logistic_regression(X_train_np, y_train)
    metrics = evaluate_model(model, X_test_np, y_test, model_name="lr_smoke")

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1
