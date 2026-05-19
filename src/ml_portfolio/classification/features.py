"""Classification feature engineering and scaling."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler


def derive_text_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create length and word-count features from text columns."""
    text_cols = ["amenities", "description", "name", "host_about", "neighborhood_overview"]
    for c in text_cols:
        if c in df.columns:
            df = df.with_columns(
                pl.col(c)
                .cast(pl.Utf8)
                .str.len_bytes()
                .alias(f"{c}_length"),
                pl.col(c)
                .cast(pl.Utf8)
                .str.split(" ")
                .list.len()
                .alias(f"{c}_word_count"),
            )
            df = df.drop(c)
    return df


def derive_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create temporal delta features from date columns."""
    if "host_since" in df.columns and "last_scraped" in df.columns:
        df = df.with_columns(
            (pl.col("last_scraped") - pl.col("host_since"))
            .dt.total_days()
            .alias("host_tenure_days")
        )

    if "first_review" in df.columns and "last_scraped" in df.columns:
        df = df.with_columns(
            (
                pl.col("last_scraped")
                - pl.col("first_review").str.to_date("%Y-%m-%d", strict=False)
            )
            .dt.total_days()
            .alias("listing_age_days")
        )

    if "last_review" in df.columns and "last_scraped" in df.columns:
        df = df.with_columns(
            (
                pl.col("last_scraped")
                - pl.col("last_review").str.to_date("%Y-%m-%d", strict=False)
            )
            .dt.total_days()
            .fill_null(0)
            .alias("days_since_last_review")
        )

    return df


def derive_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create ratio/interaction features."""
    if "bathrooms" in df.columns and "bedrooms" in df.columns:
        df = df.with_columns(
            (pl.col("bathrooms") / pl.col("bedrooms").replace(0, 1)).alias(
                "bathrooms_per_bedroom"
            )
        )

    if "beds" in df.columns and "accommodates" in df.columns:
        df = df.with_columns(
            (pl.col("beds") / pl.col("accommodates").replace(0, 1)).alias(
                "beds_per_person"
            )
        )

    if "price" in df.columns and "bedrooms" in df.columns:
        df = df.with_columns(
            (pl.col("price") / pl.col("bedrooms").replace(0, 1)).alias(
                "price_per_bedroom"
            )
        )

    return df


def aggregate_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create mean review score aggregate."""
    review_cols = [c for c in df.columns if c.startswith("review_scores_")]
    if review_cols:
        df = df.with_columns(
            pl.mean_horizontal([pl.col(c) for c in review_cols]).alias(
                "review_score_mean"
            )
        )
    return df


def encode_categoricals(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    encoders_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Encode categorical features with ordinal, one-hot, and frequency encoding.

    Returns:
        Transformed train/val/test DataFrames and an encoders metadata dict.
    """
    encoders: dict = {}

    # Helper to apply same transform to all splits
    def _apply(expr_fn):
        nonlocal df_train, df_val, df_test
        df_train = df_train.with_columns(expr_fn(df_train))
        df_val = df_val.with_columns(expr_fn(df_val))
        df_test = df_test.with_columns(expr_fn(df_test))

    # 1. Low-cardinality ordinal encoding
    ordinal_maps = {
        "host_response_time": {
            "within an hour": 4,
            "within a few hours": 3,
            "within a day": 2,
            "a few days or more": 1,
            "N/A": 0,
        },
        "room_type": {
            "Entire home/apt": 4,
            "Private room": 3,
            "Hotel room": 2,
            "Shared room": 1,
        },
    }

    for col, mapping in ordinal_maps.items():
        if col in df_train.columns:
            _apply(lambda d: pl.col(col).replace(mapping).cast(pl.Int16).alias(f"{col}_ordinal"))
            encoders[f"{col}_ordinal"] = mapping

    # Frequency-based ordinal for other low-card categoricals
    low_card_cols = [
        c for c in df_train.columns
        if df_train[c].dtype == pl.Utf8
        and 2 <= df_train[c].n_unique() <= 10
        and c not in ordinal_maps
        and c != "host_verifications"
    ]

    for col in low_card_cols:
        freq = df_train[col].value_counts().sort("count", descending=True)
        mapping = {row[col]: i for i, row in enumerate(freq.iter_rows(named=True))}
        _apply(lambda d: pl.col(col).replace(mapping).cast(pl.Int16).alias(f"{col}_ordinal"))
        encoders[f"{col}_ordinal"] = mapping

    # 2. Multi-select: host_verifications
    if "host_verifications" in df_train.columns:
        all_verifications = (
            df_train["host_verifications"]
            .cast(pl.Utf8)
            .str.split(",")
            .explode()
            .str.strip_chars()
            .unique()
            .to_list()
        )
        for v in all_verifications:
            if v:
                safe = re.sub(r"[^a-zA-Z0-9]", "_", v).lower()
                _apply(
                    lambda d: pl.col("host_verifications")
                    .cast(pl.Utf8)
                    .str.contains(v, literal=True)
                    .cast(pl.Int8)
                    .alias(f"verified_{safe}")
                )
        _apply(
            lambda d: pl.col("host_verifications")
            .cast(pl.Utf8)
            .str.split(",")
            .list.len()
            .alias("verification_count")
        )

    # 3. Medium cardinality: one-hot top 10 + other
    medium_card_cols = [
        c for c in df_train.columns
        if df_train[c].dtype == pl.Utf8
        and 10 < df_train[c].n_unique() <= 30
    ]

    for col in medium_card_cols:
        top = (
            df_train[col]
            .value_counts()
            .sort("count", descending=True)
            .head(10)[col]
            .to_list()
        )
        for val in top:
            safe = re.sub(r"[^a-zA-Z0-9]", "_", str(val)).lower()[:50]
            _apply(
                lambda d: (pl.col(col) == val)
                .cast(pl.Int8)
                .alias(f"{col}_{safe}")
            )
        _apply(
            lambda d: (~pl.col(col).is_in(top))
            .cast(pl.Int8)
            .alias(f"{col}_other")
        )

    # 4. High cardinality: frequency encoding
    high_card_cols = [
        c for c in df_train.columns
        if df_train[c].dtype == pl.Utf8
        and df_train[c].n_unique() > 30
    ]

    for col in high_card_cols:
        freq = (
            df_train.group_by(col)
            .agg(pl.count().alias(f"{col}_freq"))
        )
        freq_map = {row[col]: row[f"{col}_freq"] for row in freq.iter_rows(named=True)}
        _apply(
            lambda d: pl.col(col).replace(freq_map, default=0).cast(pl.Float64).alias(f"{col}_freq")
        )
        encoders[f"{col}_freq"] = freq_map

    # Drop original categorical columns
    cat_cols = [c for c in df_train.columns if df_train[c].dtype == pl.Utf8]
    for col in cat_cols:
        df_train = df_train.drop(col)
        df_val = df_val.drop(col)
        df_test = df_test.drop(col)

    if encoders_path is not None:
        encoders_path.parent.mkdir(parents=True, exist_ok=True)
        with open(encoders_path, "w") as f:
            # Convert sets/lists to serializable types
            serializable = {}
            for k, v in encoders.items():
                if isinstance(v, dict):
                    serializable[k] = {str(kk): vv for kk, vv in v.items()}
                else:
                    serializable[k] = v
            json.dump(serializable, f, indent=2, default=str)

    return df_train, df_val, df_test, encoders


def scale_features(
    X_train: pl.DataFrame,
    X_val: pl.DataFrame,
    X_test: pl.DataFrame,
    scaler_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, StandardScaler]:
    """Fit StandardScaler on train and transform all splits."""
    scaler = StandardScaler()
    feature_names = X_train.columns

    X_train_np = scaler.fit_transform(X_train.to_numpy())
    X_val_np = scaler.transform(X_val.to_numpy())
    X_test_np = scaler.transform(X_test.to_numpy())

    X_train_s = pl.DataFrame(X_train_np, schema=feature_names)
    X_val_s = pl.DataFrame(X_val_np, schema=feature_names)
    X_test_s = pl.DataFrame(X_test_np, schema=feature_names)

    if scaler_path is not None:
        import joblib

        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return X_train_s, X_val_s, X_test_s, scaler
