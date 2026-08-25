from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of one validation check.
    """

    name: str
    status: str
    message: str
    observed_value: int | float | str | None = None


class ValidationPipelineError(Exception):
    """
    Raised when the validation pipeline itself fails.
    """


class DataValidator:
    """
    Collection of reusable data-quality checks.

    Validation checks do not modify the DataFrame.
    """

    def __init__(
        self,
        expected_columns: list[str],
    ) -> None:

        self.expected_columns = (
            expected_columns
        )

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def check_required_columns(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        expected = set(
            self.expected_columns
        )

        actual = set(df.columns)

        missing = expected - actual

        if missing:

            return ValidationResult(
                name="Required columns",
                status="FAIL",
                message=(
                    "Missing required columns: "
                    + ", ".join(
                        sorted(missing)
                    )
                ),
                observed_value=len(missing),
            )

        return ValidationResult(
            name="Required columns",
            status="PASS",
            message=(
                "All required columns are present."
            ),
            observed_value=len(expected),
        )

    def check_unexpected_columns(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        expected = set(
            self.expected_columns
        )

        actual = set(df.columns)

        unexpected = actual - expected

        if unexpected:

            return ValidationResult(
                name="Unexpected columns",
                status="FAIL",
                message=(
                    "Unexpected columns found: "
                    + ", ".join(
                        sorted(unexpected)
                    )
                ),
                observed_value=len(
                    unexpected
                ),
            )

        return ValidationResult(
            name="Unexpected columns",
            status="PASS",
            message=(
                "No unexpected columns found."
            ),
            observed_value=0,
        )

    def check_column_order(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        actual = list(df.columns)

        if actual != self.expected_columns:

            return ValidationResult(
                name="Column order",
                status="FAIL",
                message=(
                    "Column order does not match "
                    "the expected schema."
                ),
            )

        return ValidationResult(
            name="Column order",
            status="PASS",
            message=(
                "Column order matches the schema."
            ),
        )

    def check_row_count(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        rows = len(df)

        if rows == 0:

            return ValidationResult(
                name="Row count",
                status="FAIL",
                message=(
                    "Dataset contains zero rows."
                ),
                observed_value=rows,
            )

        return ValidationResult(
            name="Row count",
            status="PASS",
            message=(
                f"Dataset contains {rows:,} rows."
            ),
            observed_value=rows,
        )

    # ------------------------------------------------------------------
    # Missing / numeric integrity
    # ------------------------------------------------------------------

    def check_missing_values(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        missing = int(
            df.isna().sum().sum()
        )

        if missing > 0:

            return ValidationResult(
                name="Missing values",
                status="FAIL",
                message=(
                    f"Dataset contains "
                    f"{missing:,} missing values."
                ),
                observed_value=missing,
            )

        return ValidationResult(
            name="Missing values",
            status="PASS",
            message=(
                "No missing values found."
            ),
            observed_value=0,
        )

    def check_infinite_values(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if numeric_df.empty:

            return ValidationResult(
                name="Infinite values",
                status="PASS",
                message=(
                    "No numeric columns require "
                    "infinite-value checking."
                ),
                observed_value=0,
            )

        infinite_count = int(
            np.isinf(
                numeric_df.to_numpy()
            ).sum()
        )

        if infinite_count > 0:

            return ValidationResult(
                name="Infinite values",
                status="FAIL",
                message=(
                    f"Found {infinite_count:,} "
                    f"infinite numeric values."
                ),
                observed_value=infinite_count,
            )

        return ValidationResult(
            name="Infinite values",
            status="PASS",
            message=(
                "No infinite numeric values found."
            ),
            observed_value=0,
        )

    # ------------------------------------------------------------------
    # Financial domain checks
    # ------------------------------------------------------------------

    def check_non_negative_values(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        columns = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]

        errors: list[str] = []
        total_invalid = 0

        for column in columns:

            if column not in df.columns:
                continue

            count = int(
                (df[column] < 0).sum()
            )

            if count > 0:

                errors.append(
                    f"{column}: "
                    f"{count:,}"
                )

                total_invalid += count

        if errors:

            return ValidationResult(
                name="Non-negative financial values",
                status="FAIL",
                message=(
                    "Negative values found: "
                    + "; ".join(errors)
                ),
                observed_value=total_invalid,
            )

        return ValidationResult(
            name="Non-negative financial values",
            status="PASS",
            message=(
                "All financial amount and balance "
                "values are non-negative."
            ),
            observed_value=0,
        )

    def check_step_values(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        if "step" not in df.columns:

            return ValidationResult(
                name="Step values",
                status="FAIL",
                message=(
                    "'step' column is missing."
                ),
            )

        invalid = int(
            (df["step"] < 0).sum()
        )

        if invalid > 0:

            return ValidationResult(
                name="Step values",
                status="FAIL",
                message=(
                    f"Found {invalid:,} "
                    f"negative step values."
                ),
                observed_value=invalid,
            )

        return ValidationResult(
            name="Step values",
            status="PASS",
            message=(
                "All step values are non-negative."
            ),
            observed_value=0,
        )

    def check_transaction_types(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        allowed = {
            "CASH_IN",
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER",
        }

        if "type" not in df.columns:

            return ValidationResult(
                name="Transaction types",
                status="FAIL",
                message=(
                    "'type' column is missing."
                ),
            )

        observed = set(
            df["type"]
            .dropna()
            .unique()
        )

        invalid = observed - allowed

        if invalid:

            return ValidationResult(
                name="Transaction types",
                status="FAIL",
                message=(
                    "Invalid transaction types: "
                    + ", ".join(
                        sorted(
                            str(value)
                            for value in invalid
                        )
                    )
                ),
                observed_value=len(invalid),
            )

        return ValidationResult(
            name="Transaction types",
            status="PASS",
            message=(
                "All transaction types are recognized."
            ),
            observed_value=len(observed),
        )

    def check_binary_flags(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        columns = [
            "isFraud",
            "isFlaggedFraud",
        ]

        errors: list[str] = []

        for column in columns:

            if column not in df.columns:
                continue

            observed = set(
                df[column]
                .dropna()
                .unique()
            )

            invalid = observed - {
                0,
                1,
            }

            if invalid:

                errors.append(
                    f"{column}: "
                    f"{sorted(invalid)}"
                )

        if errors:

            return ValidationResult(
                name="Binary fraud flags",
                status="FAIL",
                message=(
                    "Invalid binary flag values: "
                    + "; ".join(errors)
                ),
            )

        return ValidationResult(
            name="Binary fraud flags",
            status="PASS",
            message=(
                "Fraud indicators contain only 0 and 1."
            ),
        )

    # ------------------------------------------------------------------
    # Entity checks
    # ------------------------------------------------------------------

    def check_entity_identifiers(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        columns = [
            "nameOrig",
            "nameDest",
        ]

        errors: list[str] = []
        total_invalid = 0

        for column in columns:

            if column not in df.columns:
                continue

            values = (
                df[column]
                .astype("string")
            )

            empty = int(
                values.str.strip().eq("").sum()
            )

            missing = int(
                values.isna().sum()
            )

            invalid = empty + missing

            if invalid > 0:

                errors.append(
                    f"{column}: "
                    f"{invalid:,}"
                )

                total_invalid += invalid

        if errors:

            return ValidationResult(
                name="Entity identifiers",
                status="FAIL",
                message=(
                    "Empty or missing entity "
                    "identifiers: "
                    + "; ".join(errors)
                ),
                observed_value=total_invalid,
            )

        return ValidationResult(
            name="Entity identifiers",
            status="PASS",
            message=(
                "Origin and destination identifiers "
                "are populated."
            ),
            observed_value=0,
        )

    # ------------------------------------------------------------------
    # Duplicate check
    # ------------------------------------------------------------------

    def check_duplicate_rows(
        self,
        df: pd.DataFrame,
    ) -> ValidationResult:

        duplicates = int(
            df.duplicated(
                keep=False
            ).sum()
        )

        if duplicates > 0:

            return ValidationResult(
                name="Duplicate rows",
                status="WARNING",
                message=(
                    f"Found {duplicates:,} rows "
                    f"participating in duplicate groups. "
                    f"No rows were removed."
                ),
                observed_value=duplicates,
            )

        return ValidationResult(
            name="Duplicate rows",
            status="PASS",
            message=(
                "No completely duplicated rows found."
            ),
            observed_value=0,
        )

    # ------------------------------------------------------------------
    # Complete validation
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
    ) -> list[ValidationResult]:
        """
        Execute the complete validation suite.
        """

        checks: list[
            Callable[
                [pd.DataFrame],
                ValidationResult,
            ]
        ] = [
            self.check_required_columns,
            self.check_unexpected_columns,
            self.check_column_order,
            self.check_row_count,
            self.check_missing_values,
            self.check_infinite_values,
            self.check_non_negative_values,
            self.check_step_values,
            self.check_transaction_types,
            self.check_binary_flags,
            self.check_entity_identifiers,
            self.check_duplicate_rows,
        ]

        return [
            check(df)
            for check in checks
        ]