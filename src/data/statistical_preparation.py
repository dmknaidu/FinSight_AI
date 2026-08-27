from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class StatisticalPreparationError(Exception):
    """Raised when statistical preparation fails."""


CONTINUOUS_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TEMPORAL_COLUMNS = [
    "step",
]

CATEGORICAL_COLUMNS = [
    "type",
]

ENTITY_COLUMNS = [
    "nameOrig",
    "nameDest",
]

BINARY_COLUMNS = [
    "isFraud",
    "isFlaggedFraud",
]

TARGET_COLUMN = "isFraud"

ALLOWED_TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}


@dataclass(frozen=True)
class StatisticalPreparationSummary:
    rows: int
    columns: int
    fraud_transactions: int
    legitimate_transactions: int
    fraud_rate: float
    missing_values: int
    infinite_values: int


def validate_required_columns(
    df: pd.DataFrame,
) -> None:

    required = (
        CONTINUOUS_COLUMNS
        + TEMPORAL_COLUMNS
        + CATEGORICAL_COLUMNS
        + ENTITY_COLUMNS
        + BINARY_COLUMNS
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise StatisticalPreparationError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def validate_target(
    df: pd.DataFrame,
) -> None:

    values = set(
        df[TARGET_COLUMN].unique()
    )

    if not values.issubset({0, 1}):

        raise StatisticalPreparationError(
            f"Invalid {TARGET_COLUMN} values: {values}"
        )


def validate_transaction_types(
    df: pd.DataFrame,
) -> None:

    values = set(
        df["type"]
        .astype(str)
        .unique()
    )

    unexpected = (
        values
        - ALLOWED_TRANSACTION_TYPES
    )

    if unexpected:

        raise StatisticalPreparationError(
            "Unexpected transaction types: "
            + ", ".join(sorted(unexpected))
        )


def count_missing_values(
    df: pd.DataFrame,
) -> int:

    return int(
        df.isna().sum().sum()
    )


def count_infinite_values(
    df: pd.DataFrame,
) -> int:

    numeric_df = df.select_dtypes(
        include=np.number
    )

    return int(
        np.isinf(
            numeric_df.to_numpy()
        ).sum()
    )


def calculate_summary(
    df: pd.DataFrame,
) -> StatisticalPreparationSummary:

    fraud_transactions = int(
        df[TARGET_COLUMN].sum()
    )

    legitimate_transactions = (
        len(df)
        - fraud_transactions
    )

    fraud_rate = (
        fraud_transactions / len(df)
        if len(df) > 0
        else 0.0
    )

    return StatisticalPreparationSummary(
        rows=len(df),
        columns=len(df.columns),
        fraud_transactions=fraud_transactions,
        legitimate_transactions=legitimate_transactions,
        fraud_rate=fraud_rate,
        missing_values=count_missing_values(df),
        infinite_values=count_infinite_values(df),
    )


def calculate_continuous_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for column in CONTINUOUS_COLUMNS:

        series = df[column]

        records.append(
            {
                "column": column,
                "count": int(series.count()),
                "missing_count": int(
                    series.isna().sum()
                ),
                "zero_count": int(
                    (series == 0).sum()
                ),
                "zero_percentage": (
                    (series == 0).mean()
                    * 100
                ),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "std": float(series.std()),
                "variance": float(series.var()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
            }
        )

    return pd.DataFrame(records)


def calculate_type_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    profile = (
        df.groupby(
            "type",
            observed=True,
        )
        .agg(
            transactions=(
                "type",
                "size",
            ),
            fraud_transactions=(
                "isFraud",
                "sum",
            ),
        )
        .reset_index()
    )

    profile["fraud_rate"] = (
        profile["fraud_transactions"]
        / profile["transactions"]
    )

    return profile


def prepare_statistical_dataset(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    StatisticalPreparationSummary,
]:

    if df.empty:
        raise StatisticalPreparationError(
            "Canonical dataset is empty."
        )

    validate_required_columns(df)
    validate_target(df)
    validate_transaction_types(df)

    summary = calculate_summary(df)

    return (
        df.copy(),
        summary,
    )