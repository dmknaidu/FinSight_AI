from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class CanonicalValidationError(Exception):
    """Raised when canonical dataset validation fails."""


EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

EXPECTED_DTYPES = {
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


@dataclass(frozen=True)
class ValidationResult:
    name: str
    status: str
    message: str


def validate_row_count(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
) -> ValidationResult:

    if len(source_df) != len(canonical_df):

        return ValidationResult(
            "Row count preservation",
            "FAIL",
            (
                f"Source={len(source_df):,}; "
                f"Canonical={len(canonical_df):,}."
            ),
        )

    return ValidationResult(
        "Row count preservation",
        "PASS",
        f"{len(canonical_df):,} rows preserved.",
    )


def validate_column_structure(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
) -> list[ValidationResult]:

    results = []

    if list(source_df.columns) != list(
        canonical_df.columns
    ):

        results.append(
            ValidationResult(
                "Column order",
                "FAIL",
                "Canonical column order differs from source.",
            )
        )

    else:

        results.append(
            ValidationResult(
                "Column order",
                "PASS",
                "Column order matches source.",
            )
        )

    if list(canonical_df.columns) != EXPECTED_COLUMNS:

        results.append(
            ValidationResult(
                "Canonical columns",
                "FAIL",
                "Canonical columns do not match specification.",
            )
        )

    else:

        results.append(
            ValidationResult(
                "Canonical columns",
                "PASS",
                "Canonical columns match specification.",
            )
        )

    return results


def validate_dtypes(
    canonical_df: pd.DataFrame,
) -> ValidationResult:

    actual = {
        column: str(canonical_df[column].dtype)
        for column in canonical_df.columns
    }

    if actual != EXPECTED_DTYPES:

        differences = []

        for column in EXPECTED_DTYPES:

            expected = EXPECTED_DTYPES[column]
            observed = actual.get(column)

            if expected != observed:

                differences.append(
                    f"{column}: "
                    f"expected={expected}, "
                    f"actual={observed}"
                )

        return ValidationResult(
            "Canonical dtypes",
            "FAIL",
            "; ".join(differences),
        )

    return ValidationResult(
        "Canonical dtypes",
        "PASS",
        "All approved dtypes are present.",
    )


def validate_missing_values(
    canonical_df: pd.DataFrame,
) -> ValidationResult:

    missing = int(
        canonical_df.isna().sum().sum()
    )

    if missing > 0:

        return ValidationResult(
            "Missing values",
            "FAIL",
            f"{missing:,} missing values detected.",
        )

    return ValidationResult(
        "Missing values",
        "PASS",
        "No missing values detected.",
    )


def validate_financial_values(
    canonical_df: pd.DataFrame,
) -> ValidationResult:

    columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    negative_count = 0

    for column in columns:

        negative_count += int(
            (canonical_df[column] < 0).sum()
        )

    if negative_count > 0:

        return ValidationResult(
            "Financial values",
            "FAIL",
            (
                f"{negative_count:,} negative "
                "financial values detected."
            ),
        )

    return ValidationResult(
        "Financial values",
        "PASS",
        "All financial values are non-negative.",
    )


def validate_transaction_types(
    canonical_df: pd.DataFrame,
) -> ValidationResult:

    allowed = {
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    }

    observed = set(
        canonical_df["type"]
        .astype(str)
        .unique()
    )

    unexpected = observed - allowed

    if unexpected:

        return ValidationResult(
            "Transaction types",
            "FAIL",
            (
                "Unexpected transaction types: "
                + ", ".join(sorted(unexpected))
            ),
        )

    return ValidationResult(
        "Transaction types",
        "PASS",
        "All transaction types are valid.",
    )


def validate_fraud_values(
    canonical_df: pd.DataFrame,
) -> list[ValidationResult]:

    results = []

    fraud_values = set(
        canonical_df["isFraud"].unique()
    )

    flagged_values = set(
        canonical_df["isFlaggedFraud"].unique()
    )

    if not fraud_values.issubset({0, 1}):

        results.append(
            ValidationResult(
                "isFraud values",
                "FAIL",
                f"Unexpected values: {fraud_values}",
            )
        )

    else:

        results.append(
            ValidationResult(
                "isFraud values",
                "PASS",
                "Only 0 and 1 are present.",
            )
        )

    if not flagged_values.issubset({0, 1}):

        results.append(
            ValidationResult(
                "isFlaggedFraud values",
                "FAIL",
                f"Unexpected values: {flagged_values}",
            )
        )

    else:

        results.append(
            ValidationResult(
                "isFlaggedFraud values",
                "PASS",
                "Only 0 and 1 are present.",
            )
        )

    return results


def validate_fraud_preservation(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
) -> list[ValidationResult]:

    results = []

    source_fraud = int(
        source_df["isFraud"].sum()
    )

    canonical_fraud = int(
        canonical_df["isFraud"].sum()
    )

    if source_fraud != canonical_fraud:

        results.append(
            ValidationResult(
                "Fraud preservation",
                "FAIL",
                (
                    f"Source={source_fraud:,}; "
                    f"Canonical={canonical_fraud:,}."
                ),
            )
        )

    else:

        results.append(
            ValidationResult(
                "Fraud preservation",
                "PASS",
                (
                    f"{canonical_fraud:,} fraud "
                    "records preserved."
                ),
            )
        )

    source_flagged = int(
        source_df["isFlaggedFraud"].sum()
    )

    canonical_flagged = int(
        canonical_df["isFlaggedFraud"].sum()
    )

    if source_flagged != canonical_flagged:

        results.append(
            ValidationResult(
                "Flagged-fraud preservation",
                "FAIL",
                (
                    f"Source={source_flagged:,}; "
                    f"Canonical={canonical_flagged:,}."
                ),
            )
        )

    else:

        results.append(
            ValidationResult(
                "Flagged-fraud preservation",
                "PASS",
                (
                    f"{canonical_flagged:,} flagged-fraud "
                    "records preserved."
                ),
            )
        )

    return results


def validate_entity_identifiers(
    canonical_df: pd.DataFrame,
) -> ValidationResult:

    empty_orig = int(
        canonical_df["nameOrig"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_dest = int(
        canonical_df["nameDest"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_orig or empty_dest:

        return ValidationResult(
            "Entity identifiers",
            "FAIL",
            (
                f"Empty origin={empty_orig:,}; "
                f"empty destination={empty_dest:,}."
            ),
        )

    return ValidationResult(
        "Entity identifiers",
        "PASS",
        "Origin and destination identifiers are populated.",
    )


def validate_canonical_dataset(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
) -> list[ValidationResult]:

    results = []

    results.append(
        validate_row_count(
            source_df,
            canonical_df,
        )
    )

    results.extend(
        validate_column_structure(
            source_df,
            canonical_df,
        )
    )

    results.append(
        validate_dtypes(
            canonical_df
        )
    )

    results.append(
        validate_missing_values(
            canonical_df
        )
    )

    results.append(
        validate_financial_values(
            canonical_df
        )
    )

    results.append(
        validate_transaction_types(
            canonical_df
        )
    )

    results.extend(
        validate_fraud_values(
            canonical_df
        )
    )

    results.extend(
        validate_fraud_preservation(
            source_df,
            canonical_df,
        )
    )

    results.append(
        validate_entity_identifiers(
            canonical_df
        )
    )

    return results