"""Regression feature engineering and scaling."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler


def _normalise_experience(val: str | None) -> str | None:
    """Normalise inconsistent spacing in experience strings."""
    if val is None:
        return None
    # Ensure spaces around hyphens for ranges like '5-7 years'
    val = val.strip()
    import re

    val = re.sub(r"^(\d+)-(\d+)\s+years$", r"\1 - \2 years", val)
    val = re.sub(r"^(\d+)-(\d+)\s+year$", r"\1 - \2 year", val)
    return val


def apply_ordinal_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """Ordinal encoding for age, education, and experience columns."""
    age_order = [
        "under 18",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65 or over",
    ]
    edu_order = [
        "High School",
        "Some college",
        "College degree",
        "Master's degree",
        "Professional degree (MD, JD, etc.)",
        "PhD",
    ]
    exp_order = [
        "1 year or less",
        "2 - 4 years",
        "5 - 7 years",
        "8 - 10 years",
        "11 - 20 years",
        "21 - 30 years",
        "31 - 40 years",
        "41 years or more",
    ]

    mappings = {
        "age": age_order,
        "education": edu_order,
        "experience_overall_years": exp_order,
        "experience_field_years": exp_order,
    }

    for col, order in mappings.items():
        enum_dtype = pl.Enum(order)
        expr = pl.col(col).cast(pl.Utf8).str.strip_chars()
        if col in ("experience_overall_years", "experience_field_years"):
            expr = expr.map_elements(_normalise_experience, return_dtype=pl.Utf8)
        df = df.with_columns(
            expr.cast(enum_dtype)
            .to_physical()
            .cast(pl.Int32)
            .alias(f"{col}_ordinal")
        )
    return df


def apply_one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """One-hot encoding for gender and multi-hot for industry/race."""
    # Gender
    df = df.with_columns(
        (pl.col("gender") == "Man").cast(pl.Int8).alias("gender_male"),
        (pl.col("gender") == "Woman").cast(pl.Int8).alias("gender_female"),
        (pl.col("gender") == "Non-binary").cast(pl.Int8).alias("gender_non_binary"),
    )

    # Industry multi-hot
    industry_opts = [
        "Accounting",
        "Aerospace",
        "Agriculture",
        "Architecture",
        "Arts",
        "Automotive",
        "Biotechnology",
        "Business Services",
        "Chemicals",
        "Communications",
        "Computer Hardware",
        "Computer Software",
        "Construction",
        "Consulting",
        "Consumer Goods",
        "Education",
        "Energy",
        "Engineering",
        "Entertainment",
        "Environmental Services",
        "Finance",
        "Food & Beverages",
        "Government",
        "Healthcare",
        "Hospitality",
        "Human Resources",
        "Insurance",
        "Internet",
        "Legal",
        "Logistics",
        "Machinery",
        "Manufacturing",
        "Marketing",
        "Media",
        "Mining",
        "Non-Profit",
        "Pharmaceuticals",
        "Real Estate",
        "Retail",
        "Security",
        "Sports",
        "Technology",
        "Telecommunications",
        "Transportation",
        "Utilities",
    ]
    for opt in industry_opts:
        safe = re.sub(r"[^a-zA-Z0-9]", "_", opt).lower()
        df = df.with_columns(
            pl.col("industry")
            .str.contains(re.escape(opt), literal=False)
            .cast(pl.Int8)
            .alias(f"industry_{safe}")
        )

    # Race multi-hot
    race_opts = [
        "Asian",
        "Black",
        "Hispanic",
        "Middle Eastern",
        "Native American",
        "Pacific Islander",
        "White",
        "Another race",
    ]
    for opt in race_opts:
        safe = re.sub(r"[^a-zA-Z0-9]", "_", opt).lower()
        df = df.with_columns(
            pl.col("race")
            .str.contains(re.escape(opt), literal=False)
            .cast(pl.Int8)
            .alias(f"race_{safe}")
        )

    return df


def extract_job_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extract seniority level from job title via regex."""
    patterns = {
        "job_junior": re.compile(r"(?i)(intern|junior|entry|associate|assistant)"),
        "job_senior": re.compile(r"(?i)(senior|lead|principal|staff|architect)"),
        "job_manager": re.compile(r"(?i)(manager|mgr)"),
        "job_director": re.compile(r"(?i)(director|dir\.?)"),
        "job_vp": re.compile(r"(?i)(vp|vice president)"),
        "job_c_level": re.compile(r"(?i)(ceo|cto|cfo|cio|coo|chief)"),
    }

    for col_name, pattern in patterns.items():
        df = df.with_columns(
            pl.col("job")
            .cast(pl.Utf8)
            .str.contains(pattern.pattern)
            .cast(pl.Int8)
            .alias(col_name)
        )

    # Default 'other' if none matched
    flag_cols = list(patterns.keys())
    df = df.with_columns(
        (1 - pl.sum_horizontal([pl.col(c) for c in flag_cols])).clip(0, 1).alias("job_other")
    )

    return df


def extract_timestamp_features(df: pl.DataFrame) -> pl.DataFrame:
    """Parse timestamp and extract temporal components."""
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False)
    ).with_columns(
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        pl.col("timestamp").dt.hour().alias("hour"),
    )
    return df


def create_composite_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create interaction and ratio features."""
    df = df.with_columns(
        (
            pl.col("experience_field_years_ordinal")
            / pl.col("experience_overall_years_ordinal").replace(0, 1)
        ).alias("experience_ratio"),
        (
            pl.col("education_ordinal") * pl.col("experience_overall_years_ordinal")
        ).alias("education_experience_interaction"),
        (pl.col("experience_overall_years_ordinal") ** 2).alias(
            "experience_overall_sq"
        ),
    )
    return df


def drop_original_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop raw categorical/text columns after encoding."""
    to_drop = [
        "job_context",
        "income_context",
        "currency",
        "currency_other",
        "country",
        "state",
        "city",
        "is_us",
        "salary",
        "compensation",
        "timestamp",
        "job",
        "industry",
        "race",
        "gender",
        "age",
        "education",
        "experience_overall_years",
        "experience_field_years",
    ]
    existing = [c for c in to_drop if c in df.columns]
    return df.drop(existing)


def scale_features(
    X_train: pl.DataFrame,
    X_val: pl.DataFrame,
    X_test: pl.DataFrame,
    scaler_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, StandardScaler]:
    """Fit StandardScaler on training data and transform all splits.

    Optionally persists scaler and metadata to disk.
    """
    scaler = StandardScaler()
    feature_names = X_train.columns

    X_train_np = scaler.fit_transform(X_train.to_numpy())
    X_val_np = scaler.transform(X_val.to_numpy())
    X_test_np = scaler.transform(X_test.to_numpy())

    X_train_scaled = pl.DataFrame(X_train_np, schema=feature_names)
    X_val_scaled = pl.DataFrame(X_val_np, schema=feature_names)
    X_test_scaled = pl.DataFrame(X_test_np, schema=feature_names)

    if scaler_path is not None:
        import joblib

        joblib.dump(scaler, scaler_path)

    if metadata_path is not None:
        metadata = {
            "feature_names": list(feature_names),
            "means": scaler.mean_.tolist(),
            "scales": scaler.scale_.tolist(),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
