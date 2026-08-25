from __future__ import annotations

from pathlib import Path

import pandas as pd


class CanonicalDatasetError(Exception):
    """Raised when canonical dataset generation fails."""


APPROVED_DTYPES = {
    "step": "uint16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "object",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "object",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "uint8",
    "isFlaggedFraud": "uint8",
}


EXPECTED_COLUMNS = list(
    APPROVED_DTYPES.keys()
)


def optimize_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply only the dtype transformations approved
    during Phase 1.
    """

    result = df.copy()

    result["step"] = (
        result["step"].astype("uint16")
    )

    result["type"] = (
        result["type"].astype("category")
    )

    result["isFraud"] = (
        result["isFraud"].astype("uint8")
    )

    result["isFlaggedFraud"] = (
        result["isFlaggedFraud"].astype("uint8")
    )

    return result


def validate_canonical_structure(
    df: pd.DataFrame,
) -> None:
    """Validate canonical dataset structure."""

    actual_columns = list(
        df.columns
    )

    if actual_columns != EXPECTED_COLUMNS:
        raise CanonicalDatasetError(
            "Canonical columns do not match "
            "the approved specification."
        )

    for column, expected_dtype in (
        APPROVED_DTYPES.items()
    ):

        actual_dtype = str(
            df[column].dtype
        )

        if actual_dtype != expected_dtype:
            raise CanonicalDatasetError(
                f"Invalid dtype for {column}: "
                f"expected={expected_dtype}, "
                f"actual={actual_dtype}"
            )


def validate_canonical_values(
    df: pd.DataFrame,
) -> None:
    """Validate canonical dataset values."""

    if df.isna().any().any():
        raise CanonicalDatasetError(
            "Canonical dataset contains "
            "missing values."
        )

    financial_columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    for column in financial_columns:

        if (df[column] < 0).any():

            raise CanonicalDatasetError(
                f"Negative values found in {column}."
            )

    allowed_types = {
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    }

    observed_types = set(
        df["type"]
        .astype(str)
        .unique()
    )

    unexpected_types = (
        observed_types
        - allowed_types
    )

    if unexpected_types:

        raise CanonicalDatasetError(
            "Unexpected transaction types: "
            + ", ".join(
                sorted(unexpected_types)
            )
        )

    if not set(
        df["isFraud"].unique()
    ).issubset({0, 1}):

        raise CanonicalDatasetError(
            "Invalid isFraud values."
        )

    if not set(
        df["isFlaggedFraud"].unique()
    ).issubset({0, 1}):

        raise CanonicalDatasetError(
            "Invalid isFlaggedFraud values."
        )


def generate_canonical_dataset(
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the canonical dataset from the validated
    source dataframe.

    No analytical features are introduced.
    """

    if source_df.empty:

        raise CanonicalDatasetError(
            "Source dataset is empty."
        )

    canonical_df = optimize_dtypes(
        source_df
    )

    validate_canonical_structure(
        canonical_df
    )

    validate_canonical_values(
        canonical_df
    )

    return canonical_df


def persist_canonical_dataset(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Persist the canonical dataset as Parquet."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        df.to_parquet(
            output_path,
            index=False,
            compression="snappy",
        )

    except Exception as exc:

        raise CanonicalDatasetError(
            f"Failed to persist canonical dataset: "
            f"{exc}"
        ) from exc