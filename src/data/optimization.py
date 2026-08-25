from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class OptimizationError(Exception):
    """Raised when safe dtype optimization fails."""


@dataclass(frozen=True)
class OptimizationResult:
    """Metadata describing the optimization result."""

    original_memory_bytes: int
    optimized_memory_bytes: int
    memory_saved_bytes: int
    original_memory_mb: float
    optimized_memory_mb: float
    memory_saved_mb: float
    memory_reduction_percentage: float


OPTIMIZATION_RULES = {
    "step": "uint16",
    "type": "category",
    "isFraud": "uint8",
    "isFlaggedFraud": "uint8",
}


def calculate_memory_bytes(
    df: pd.DataFrame,
) -> int:
    """Return deep memory usage of a DataFrame."""

    return int(
        df.memory_usage(
            index=False,
            deep=True,
        ).sum()
    )


def optimize_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply only the dtype optimizations approved by
    the Step 3A memory analysis.

    The input DataFrame is not modified.
    """

    optimized = df.copy()

    try:
        optimized["step"] = (
            optimized["step"]
            .astype("uint16")
        )

        optimized["type"] = (
            optimized["type"]
            .astype("category")
        )

        optimized["isFraud"] = (
            optimized["isFraud"]
            .astype("uint8")
        )

        optimized["isFlaggedFraud"] = (
            optimized["isFlaggedFraud"]
            .astype("uint8")
        )

    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizationError(
            "Safe dtype optimization failed."
        ) from exc

    return optimized


def validate_optimized_dtypes(
    df: pd.DataFrame,
) -> None:
    """
    Verify that the approved dtype changes were applied
    and that unrelated columns retained their dtypes.
    """

    expected_dtypes = {
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

    errors: list[str] = []

    for column, expected_dtype in expected_dtypes.items():

        actual_dtype = str(
            df[column].dtype
        )

        if actual_dtype != expected_dtype:
            errors.append(
                f"{column}: expected "
                f"{expected_dtype}, got "
                f"{actual_dtype}"
            )

    if errors:
        raise OptimizationError(
            "Optimized dtype validation failed:\n"
            + "\n".join(errors)
        )


def validate_value_ranges(
    original: pd.DataFrame,
    optimized: pd.DataFrame,
) -> None:
    """
    Verify that the optimized integer columns preserve
    their original minimum and maximum values.
    """

    columns = [
        "step",
        "isFraud",
        "isFlaggedFraud",
    ]

    errors: list[str] = []

    for column in columns:

        original_min = original[column].min()
        optimized_min = optimized[column].min()

        original_max = original[column].max()
        optimized_max = optimized[column].max()

        if original_min != optimized_min:
            errors.append(
                f"{column}: minimum changed "
                f"from {original_min} "
                f"to {optimized_min}"
            )

        if original_max != optimized_max:
            errors.append(
                f"{column}: maximum changed "
                f"from {original_max} "
                f"to {optimized_max}"
            )

    if errors:
        raise OptimizationError(
            "Value-range validation failed:\n"
            + "\n".join(errors)
        )


def validate_row_and_column_structure(
    original: pd.DataFrame,
    optimized: pd.DataFrame,
) -> None:
    """
    Verify that optimization does not change row count,
    column count, names, or ordering.
    """

    errors: list[str] = []

    if len(original) != len(optimized):
        errors.append(
            f"Row count changed: "
            f"{len(original)} → {len(optimized)}"
        )

    if len(original.columns) != len(
        optimized.columns
    ):
        errors.append(
            f"Column count changed: "
            f"{len(original.columns)} → "
            f"{len(optimized.columns)}"
        )

    if list(original.columns) != list(
        optimized.columns
    ):
        errors.append(
            "Column names or ordering changed."
        )

    if errors:
        raise OptimizationError(
            "Structural validation failed:\n"
            + "\n".join(errors)
        )


def validate_logical_equivalence(
    original: pd.DataFrame,
    optimized: pd.DataFrame,
) -> None:
    """
    Verify that all logical values remain unchanged.

    The comparison is performed column-by-column so that
    categorical storage does not cause a false mismatch.
    """

    if list(original.columns) != list(
        optimized.columns
    ):
        raise OptimizationError(
            "Cannot compare DataFrames with "
            "different column structures."
        )

    errors: list[str] = []

    for column in original.columns:

        original_values = (
            original[column]
            .astype("string")
        )

        optimized_values = (
            optimized[column]
            .astype("string")
        )

        equal = (
            original_values
            .eq(optimized_values)
        )

        mismatch_count = int(
            (~equal).sum()
        )

        if mismatch_count > 0:
            errors.append(
                f"{column}: "
                f"{mismatch_count:,} values differ"
            )

    if errors:
        raise OptimizationError(
            "Logical equivalence validation failed:\n"
            + "\n".join(errors)
        )


def calculate_optimization_result(
    original: pd.DataFrame,
    optimized: pd.DataFrame,
) -> OptimizationResult:
    """Calculate memory savings after optimization."""

    original_bytes = calculate_memory_bytes(
        original
    )

    optimized_bytes = calculate_memory_bytes(
        optimized
    )

    saved_bytes = (
        original_bytes
        - optimized_bytes
    )

    reduction_percentage = (
        saved_bytes
        / original_bytes
        * 100
        if original_bytes
        else 0.0
    )

    return OptimizationResult(
        original_memory_bytes=original_bytes,
        optimized_memory_bytes=optimized_bytes,
        memory_saved_bytes=saved_bytes,
        original_memory_mb=(
            original_bytes
            / (1024 ** 2)
        ),
        optimized_memory_mb=(
            optimized_bytes
            / (1024 ** 2)
        ),
        memory_saved_mb=(
            saved_bytes
            / (1024 ** 2)
        ),
        memory_reduction_percentage=(
            reduction_percentage
        ),
    )


def validate_optimization(
    original: pd.DataFrame,
    optimized: pd.DataFrame,
) -> None:
    """
    Run the complete optimization validation suite.
    """

    validate_row_and_column_structure(
        original,
        optimized,
    )

    validate_optimized_dtypes(
        optimized
    )

    validate_value_ranges(
        original,
        optimized,
    )

    validate_logical_equivalence(
        original,
        optimized,
    )