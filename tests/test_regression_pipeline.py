"""Smoke tests for the regression pipeline."""

from __future__ import annotations

import numpy as np

from ml_portfolio.regression.features import scale_features
from ml_portfolio.regression.models import evaluate_model, train_ridge
from ml_portfolio.regression.pipeline import engineer_features
from ml_portfolio.regression.preprocessing import (
    apply_log_transform,
    clean_data,
    load_raw_data,
    split_data,
)
from ml_portfolio.shared.utils import get_default_paths


def test_load_raw_data() -> None:
    paths = get_default_paths("regression")
    df = load_raw_data(paths["raw_dir"] / "salary-survey.csv")
    assert len(df) > 0
    assert "Timestamp" in df.columns


def test_clean_data() -> None:
    paths = get_default_paths("regression")
    df = load_raw_data(paths["raw_dir"] / "salary-survey.csv")
    df_clean = clean_data(df)
    assert "annual_salary" in df_clean.columns
    assert df_clean["annual_salary"].min() >= 15_000
    assert df_clean["annual_salary"].max() <= 500_000
    assert df_clean["currency"].unique().to_list() == ["USD"]


def test_split_data() -> None:
    paths = get_default_paths("regression")
    df = load_raw_data(paths["raw_dir"] / "salary-survey.csv")
    df_clean = clean_data(df)
    df_clean = apply_log_transform(df_clean)
    train, val, test = split_data(df_clean)

    total = len(train) + len(val) + len(test)
    assert total == len(df_clean)
    # Rough 70/15/15 split
    assert 0.65 <= len(train) / total <= 0.75
    assert 0.10 <= len(val) / total <= 0.20
    assert 0.10 <= len(test) / total <= 0.20


def test_feature_engineering() -> None:
    paths = get_default_paths("regression")
    df = load_raw_data(paths["raw_dir"] / "salary-survey.csv")
    df_clean = clean_data(df)
    df_enc = engineer_features(df_clean)

    assert "age_ordinal" in df_enc.columns
    assert "job_senior" in df_enc.columns
    assert "experience_ratio" in df_enc.columns


def test_scale_features() -> None:
    paths = get_default_paths("regression")
    df = load_raw_data(paths["raw_dir"] / "salary-survey.csv")
    df_clean = clean_data(df)
    df_clean = apply_log_transform(df_clean)
    train, val, test = split_data(df_clean)

    train = engineer_features(train)
    val = engineer_features(val)
    test = engineer_features(test)

    y_train = train["log_salary"].to_numpy()
    X_train = train.drop("log_salary")
    X_val = val.drop("log_salary")
    X_test = test.drop("log_salary")

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    assert X_train_s.shape[1] == X_val_s.shape[1] == X_test_s.shape[1]
    # After scaling, mean should be ~0 for train
    np.testing.assert_allclose(X_train_s.to_numpy().mean(axis=0), 0, atol=1e-6)


def test_train_and_evaluate_ridge() -> None:
    paths = get_default_paths("regression")
    df = load_raw_data(paths["raw_dir"] / "salary-survey.csv")
    df_clean = clean_data(df)
    df_clean = apply_log_transform(df_clean)
    train, val, test = split_data(df_clean)

    train = engineer_features(train)
    test = engineer_features(test)

    y_train = train["log_salary"].to_numpy()
    y_test = test["log_salary"].to_numpy()
    X_train = train.drop("log_salary").to_numpy()
    X_test = test.drop("log_salary").to_numpy()

    model = train_ridge(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, model_name="ridge_smoke")

    assert "rmse_log" in metrics
    assert "r2_log" in metrics
    assert metrics["r2_log"] > 0  # Should explain some variance
