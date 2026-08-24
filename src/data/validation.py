from dataclasses import dataclass
from typing import Any

import pandas as pd


EXPECTED_TRANSACTION_TYPES = {
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "DEBIT",
}


@dataclass
class ValidationResult:
    check: str
    status: str
    value: Any
    message: str


def validate_schema(df: pd.DataFrame) -> list[ValidationResult]:
    results = []

    required_columns = {
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
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        results.append(
            ValidationResult(
                check="Required columns",
                status="FAIL",
                value=len(missing_columns),
                message=(
                    "Missing columns: "
                    + ", ".join(sorted(missing_columns))
                ),
            )
        )
    else:
        results.append(
            ValidationResult(
                check="Required columns",
                status="PASS",
                value=len(required_columns),
                message="All required PaySim columns are present.",
            )
        )

    return results


def validate_missing_values(df: pd.DataFrame) -> list[ValidationResult]:
    results = []

    missing_total = int(df.isna().sum().sum())

    if missing_total == 0:
        results.append(
            ValidationResult(
                check="Missing values",
                status="PASS",
                value=missing_total,
                message="No missing values found.",
            )
        )
    else:
        missing_columns = (
            df.isna()
            .sum()
            .loc[lambda x: x > 0]
            .sort_values(ascending=False)
            .to_dict()
        )

        results.append(
            ValidationResult(
                check="Missing values",
                status="WARN",
                value=missing_total,
                message=f"Missing values by column: {missing_columns}",
            )
        )

    return results


def validate_duplicates(df: pd.DataFrame) -> list[ValidationResult]:
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        status = "PASS"
        message = "No completely duplicated rows found."
    else:
        status = "WARN"
        message = f"Found {duplicate_count:,} duplicated rows."

    return [
        ValidationResult(
            check="Duplicate rows",
            status=status,
            value=duplicate_count,
            message=message,
        )
    ]


def validate_transaction_types(
    df: pd.DataFrame,
) -> list[ValidationResult]:
    observed_types = set(df["type"].dropna().unique())

    unexpected_types = observed_types - EXPECTED_TRANSACTION_TYPES

    if unexpected_types:
        return [
            ValidationResult(
                check="Transaction types",
                status="WARN",
                value=sorted(unexpected_types),
                message=(
                    "Unexpected transaction types detected: "
                    + ", ".join(sorted(unexpected_types))
                ),
            )
        ]

    return [
        ValidationResult(
            check="Transaction types",
            status="PASS",
            value=sorted(observed_types),
            message="All transaction types are recognized.",
        )
    ]


def validate_numeric_values(
    df: pd.DataFrame,
) -> list[ValidationResult]:
    results = []

    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    for column in numeric_columns:
        negative_count = int((df[column] < 0).sum())

        if negative_count == 0:
            results.append(
                ValidationResult(
                    check=f"Negative values: {column}",
                    status="PASS",
                    value=negative_count,
                    message=f"No negative values found in {column}.",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check=f"Negative values: {column}",
                    status="WARN",
                    value=negative_count,
                    message=(
                        f"{negative_count:,} negative values found "
                        f"in {column}."
                    ),
                )
            )

    return results


def validate_target_values(
    df: pd.DataFrame,
) -> list[ValidationResult]:
    results = []

    observed_fraud_values = set(df["isFraud"].dropna().unique())
    observed_flag_values = set(df["isFlaggedFraud"].dropna().unique())

    expected_binary_values = {0, 1}

    if observed_fraud_values.issubset(expected_binary_values):
        results.append(
            ValidationResult(
                check="isFraud values",
                status="PASS",
                value=sorted(observed_fraud_values),
                message="isFraud contains only binary values.",
            )
        )
    else:
        results.append(
            ValidationResult(
                check="isFraud values",
                status="FAIL",
                value=sorted(observed_fraud_values),
                message="Unexpected values detected in isFraud.",
            )
        )

    if observed_flag_values.issubset(expected_binary_values):
        results.append(
            ValidationResult(
                check="isFlaggedFraud values",
                status="PASS",
                value=sorted(observed_flag_values),
                message="isFlaggedFraud contains only binary values.",
            )
        )
    else:
        results.append(
            ValidationResult(
                check="isFlaggedFraud values",
                status="FAIL",
                value=sorted(observed_flag_values),
                message="Unexpected values detected in isFlaggedFraud.",
            )
        )

    return results


def validate_dataset(df: pd.DataFrame) -> list[ValidationResult]:
    """
    Run all structural and basic data-quality validations.
    """

    results = []

    results.extend(validate_schema(df))
    results.extend(validate_missing_values(df))
    results.extend(validate_duplicates(df))
    results.extend(validate_transaction_types(df))
    results.extend(validate_numeric_values(df))
    results.extend(validate_target_values(df))

    return results