"""Regression data preprocessing: load, clean, split."""

from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ml_portfolio.shared.utils import RANDOM_STATE

SALARY_MIN = 15_000
SALARY_MAX = 500_000
CURRENCY_FILTER = "USD"

COLUMN_MAP = {
    "How old are you?": "age",
    "What industry do you work in?": "industry",
    "Job title": "job",
    "If your job title needs additional context, please clarify here:": "job_context",
    "What is your annual salary? (You'll indicate the currency in a later question. If you are part-time or hourly, please enter an annualized equivalent -- what you would earn if you worked the job 40 hours a week, 52 weeks a year.)": "salary",
    "How much additional monetary compensation do you get, if any (for example, bonuses or overtime in an average year)? Please only include monetary compensation here, not the value of benefits.": "compensation",
    "Please indicate the currency": "currency",
    'If ""Other,"" please indicate the currency here: ': "currency_other",
    "If your income needs additional context, please provide it here:": "income_context",
    "What country do you work in?": "country",
    "If you're in the U.S., what state do you work in?": "state",
    "What city do you work in?": "city",
    "How many years of professional work experience do you have overall?": "experience_overall_years",
    "How many years of professional work experience do you have in your field?": "experience_field_years",
    "What is your highest level of education completed?": "education",
    "What is your gender?": "gender",
    "What is your race? (Choose all that apply.)": "race",
    "Timestamp": "timestamp",
}


def load_raw_data(raw_path: Path) -> pl.DataFrame:
    """Load the raw salary survey CSV."""
    return pl.read_csv(raw_path, infer_schema_length=10000)


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clean and filter the raw salary survey data.

    Steps:
        1. Rename columns to short names.
        2. Compute annual_salary = salary + compensation.
        3. Filter to USD currency and valid salary range.
        4. Fuzzy-match country to "united states" (strictness 0.6).
        5. Drop rows with any remaining nulls in core columns.
        6. Drop "Prefer not to answer" responses.
    """
    df = df.rename(COLUMN_MAP)

    # Compute annual salary
    df = df.with_columns(
        pl.col("salary")
        .cast(pl.Utf8)
        .str.replace_all(",", "")
        .cast(pl.Float64),
        pl.col("compensation").fill_null(0).cast(pl.Float64),
    ).with_columns(
        (pl.col("salary") + pl.col("compensation")).alias("annual_salary"),
    )

    # Filter currency and salary range
    df = df.filter(
        (pl.col("currency") == CURRENCY_FILTER)
        & (pl.col("annual_salary") >= SALARY_MIN)
        & (pl.col("annual_salary") <= SALARY_MAX)
    )

    # Fuzzy-match US country
    target = "united states"
    threshold = 0.6

    def _is_us(country: str | None) -> bool:
        if country is None:
            return False
        words = country.lower().split()
        return any(
            difflib.SequenceMatcher(None, w, target).ratio() >= threshold
            for w in words
        )

    df = df.with_columns(
        pl.col("country")
        .map_elements(_is_us, return_dtype=pl.Boolean)
        .alias("is_us")
    ).filter(pl.col("is_us"))

    # Drop rows with nulls in core columns
    core_cols = [
        "age",
        "industry",
        "job",
        "annual_salary",
        "experience_overall_years",
        "experience_field_years",
        "education",
        "gender",
        "race",
    ]
    df = df.drop_nulls(subset=core_cols)

    # Drop "Prefer not to answer"
    for col in ["gender", "race"]:
        df = df.filter(~pl.col(col).str.contains("(?i)prefer not to answer"))

    return df


def apply_log_transform(df: pl.DataFrame) -> pl.DataFrame:
    """Apply log1p transform to annual_salary."""
    return df.with_columns(
        pl.col("annual_salary").log1p().alias("log_salary")
    )


def split_data(
    df: pl.DataFrame,
    target_col: str = "log_salary",
    random_state: int = RANDOM_STATE,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Stratified 70/15/15 split on binned target values.

    Returns:
        train, val, test DataFrames.
    """
    y = df[target_col].to_numpy()
    # Create 20 stratification bins
    bins = np.linspace(y.min(), y.max(), num=20)
    stratify_labels = np.digitize(y, bins=bins)
    # Merge edge bins that may have < 2 members
    stratify_labels = np.clip(stratify_labels, 1, len(bins) - 1)

    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=stratify_labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=stratify_labels[temp_idx],
        random_state=random_state,
    )

    train_df = df[train_idx]
    val_df = df[val_idx]
    test_df = df[test_idx]

    return train_df, val_df, test_df
