"""Classification data preprocessing: load, clean, split, balance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ml_portfolio.shared.utils import RANDOM_STATE

TARGET_COL = "host_is_superhost"
DROP_COLS = [
    "estimated_revenue_l365d",
    "availability_eoy",
    "number_of_reviews_ly",
    "estimated_occupancy_l365d",
]


def load_raw_data(raw_dir: Path) -> pl.DataFrame:
    """Load and concatenate the four Airbnb timepoint CSVs."""
    dfs = []
    for i in range(1, 5):
        path = raw_dir / f"airbnb_timepoint{i}.csv"
        dfs.append(pl.read_csv(path, infer_schema_length=10000))
    df = pl.concat(dfs, how="diagonal_relaxed", rechunk=True)
    df = df.drop([c for c in DROP_COLS if c in df.columns])
    return df


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clean and impute the raw Airbnb listings data.

    Steps:
        1. Drop rows with null target.
        2. Drop uninformative columns (URLs, IDs, 100% null).
        3. Drop rows missing values in low-null core columns.
        4. Impute remaining nulls by column type.
        5. Parse dates and booleans.
    """
    # Drop null targets
    df = df.drop_nulls(subset=[TARGET_COL])

    # Drop uninformative columns
    uninformative = [
        "calendar_updated",
        "listing_url",
        "picture_url",
        "host_url",
        "host_thumbnail_url",
        "host_picture_url",
        "license",
        "host_total_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "scrape_id",
        "source",
        "last_searched",
    ]
    df = df.drop([c for c in uninformative if c in df.columns])

    # Low-null columns: drop any row with null in these
    low_null_cols = [
        "description",
        "host_name",
        "host_response_time",
        "bathrooms_text",
        "minimum_nights",
        "maximum_nights",
        "minimum_minimum_nights",
        "maximum_minimum_nights",
        "minimum_maximum_nights",
        "maximum_maximum_nights",
        "minimum_nights_avg_ntm",
        "maximum_nights_avg_ntm",
    ]
    existing_low = [c for c in low_null_cols if c in df.columns]
    df = df.drop_nulls(subset=existing_low)

    # Imputation strategies
    # Text columns -> empty string
    text_cols = ["neighborhood_overview", "neighbourhood"]
    for c in text_cols:
        if c in df.columns:
            df = df.with_columns(pl.col(c).fill_null(""))

    # Categorical -> "Unknown"
    cat_cols = ["host_location", "host_neighbourhood"]
    for c in cat_cols:
        if c in df.columns:
            df = df.with_columns(pl.col(c).fill_null("Unknown"))

    # Boolean mapping
    bool_cols = [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "has_availability",
        "instant_bookable",
    ]
    for c in bool_cols:
        if c in df.columns:
            df = df.with_columns(
                (pl.col(c) == "t").alias(c)
            )

    # Price parsing
    if "price" in df.columns:
        df = df.with_columns(
            pl.col("price")
            .cast(pl.Utf8)
            .str.replace_all("[$,]", "")
            .cast(pl.Float64, strict=False)
            .alias("price")
        )

    # Acceptance / response rate parsing
    for col in ["host_acceptance_rate", "host_response_rate"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .str.replace_all("[%]", "")
                .cast(pl.Float64, strict=False)
                .alias(col)
            )

    # Forward-fill review columns by id
    review_cols = [c for c in df.columns if c.startswith("review_scores_") or c == "reviews_per_month"]
    if review_cols:
        df = df.sort(["id", "last_scraped"])
        df = df.with_columns(
            [pl.col(c).forward_fill().over("id") for c in review_cols]
        )

    # Create has_* flags for review columns then median impute
    for c in review_cols:
        df = df.with_columns(
            pl.col(c).is_not_null().cast(pl.Int8).alias(f"has_{c}")
        )
        median = df[c].median()
        df = df.with_columns(pl.col(c).fill_null(median))

    # Numeric median impute for beds, bathrooms, bedrooms, price
    for col in ["beds", "bathrooms", "bedrooms", "price"]:
        if col in df.columns:
            median = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median))

    # host_acceptance_rate / host_response_rate flags + median
    for col in ["host_acceptance_rate", "host_response_rate"]:
        if col in df.columns:
            flag_name = f"has_{col.split('_')[1]}_history"
            df = df.with_columns(
                pl.col(col).is_not_null().cast(pl.Int8).alias(flag_name)
            )
            median = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median))

    # has_availability fill null -> False
    if "has_availability" in df.columns:
        df = df.with_columns(pl.col("has_availability").fill_null(False))

    # Parse dates
    for col in ["last_scraped", "host_since"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).str.to_date("%Y-%m-%d", strict=False))

    return df


def split_data(
    df: pl.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """ID-stratified 70/15/15 split ensuring no temporal leakage.

    Uses the *last known* superhost status per id as the stratification label.
    """
    id_target = (
        df.sort("last_scraped")
        .group_by("id")
        .agg(pl.col(TARGET_COL).last())
    )

    unique_ids = id_target["id"].to_numpy()
    stratify = id_target[TARGET_COL].to_numpy()

    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=0.30,
        stratify=stratify,
        random_state=random_state,
    )
    temp_target = id_target.filter(pl.col("id").is_in(temp_ids))[TARGET_COL].to_numpy()
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        stratify=temp_target,
        random_state=random_state,
    )

    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    test_ids_set = set(test_ids)

    assert len(train_ids_set & val_ids_set) == 0, "Train-Val ID overlap"
    assert len(train_ids_set & test_ids_set) == 0, "Train-Test ID overlap"
    assert len(val_ids_set & test_ids_set) == 0, "Val-Test ID overlap"

    train_df = df.filter(pl.col("id").is_in(train_ids_set))
    val_df = df.filter(pl.col("id").is_in(val_ids_set))
    test_df = df.filter(pl.col("id").is_in(test_ids_set))

    return train_df, val_df, test_df


def balance_train_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    categorical_indices: list[int],
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance the training set with SMOTENC + RandomUnderSampler.

    Args:
        X_train: Feature matrix (numpy).
        y_train: Target vector (numpy).
        categorical_indices: Column indices of categorical features.
        random_state: Reproducibility seed.

    Returns:
        Resampled X_train, y_train.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC
    from imblearn.under_sampling import RandomUnderSampler

    pipeline = ImbPipeline(
        [
            (
                "smotenc",
                SMOTENC(
                    categorical_features=categorical_indices,
                    sampling_strategy=0.5,
                    random_state=random_state,
                    k_neighbors=5,
                ),
            ),
            (
                "undersampler",
                RandomUnderSampler(
                    sampling_strategy=0.75,
                    random_state=random_state,
                ),
            ),
        ]
    )

    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    return X_res, y_res
