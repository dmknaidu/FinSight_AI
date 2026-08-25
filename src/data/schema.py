from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class SchemaError(Exception):
    """
    Raised when the dataset schema contract is invalid.
    """


def load_schema(schema_path: str | Path) -> dict[str, Any]:
    """
    Load the schema contract from a YAML file.
    """

    path = Path(schema_path)

    if not path.exists():
        raise SchemaError(
            f"Schema file does not exist: {path.resolve()}"
        )

    if not path.is_file():
        raise SchemaError(
            f"Schema path is not a file: {path.resolve()}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            schema = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SchemaError(
            f"Invalid YAML schema file: {path.resolve()}"
        ) from exc

    if not isinstance(schema, dict):
        raise SchemaError(
            "Schema file must contain a YAML mapping."
        )

    if "columns" not in schema:
        raise SchemaError(
            "Schema must contain a 'columns' section."
        )

    if not isinstance(schema["columns"], dict):
        raise SchemaError(
            "'columns' section must be a mapping."
        )

    return schema


def validate_schema_definition(
    schema: dict[str, Any],
) -> None:
    """
    Validate the structure of the schema configuration itself.
    """

    columns = schema.get("columns", {})

    if not columns:
        raise SchemaError(
            "Schema contains no column definitions."
        )

    for column_name, definition in columns.items():

        if not isinstance(definition, dict):
            raise SchemaError(
                f"Definition for '{column_name}' must be a mapping."
            )

        if "dtype" not in definition:
            raise SchemaError(
                f"Column '{column_name}' is missing 'dtype'."
            )

        if "nullable" not in definition:
            raise SchemaError(
                f"Column '{column_name}' is missing 'nullable'."
            )


def get_required_columns(
    schema: dict[str, Any],
) -> set[str]:
    """
    Return the columns required by the schema contract.
    """

    return set(
        schema["columns"].keys()
    )


def validate_columns(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> None:
    """
    Validate that the DataFrame contains exactly the expected
    schema columns.

    Column order is treated separately from column membership.
    """

    expected = get_required_columns(schema)
    actual = set(df.columns)

    missing = expected - actual
    unexpected = actual - expected

    errors: list[str] = []

    if missing:
        errors.append(
            "Missing columns: "
            + ", ".join(sorted(missing))
        )

    if unexpected:
        errors.append(
            "Unexpected columns: "
            + ", ".join(sorted(unexpected))
        )

    if errors:
        raise SchemaError(
            "Column schema validation failed:\n"
            + "\n".join(errors)
        )


def validate_column_order(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> None:
    """
    Validate the expected column order.

    PaySim's raw dataset has a known ordering, so we preserve
    and explicitly verify it.
    """

    expected = list(
        schema["columns"].keys()
    )

    actual = list(df.columns)

    if actual != expected:
        raise SchemaError(
            "Column order mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def validate_nullability(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> None:
    """
    Validate NULL constraints.
    """

    errors: list[str] = []

    for column_name, definition in schema["columns"].items():

        nullable = definition.get(
            "nullable",
            True,
        )

        if not nullable:

            null_count = int(
                df[column_name].isna().sum()
            )

            if null_count > 0:
                errors.append(
                    f"{column_name}: "
                    f"{null_count:,} null values"
                )

    if errors:
        raise SchemaError(
            "Nullability validation failed:\n"
            + "\n".join(errors)
        )


def validate_allowed_values(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> None:
    """
    Validate categorical/enumerated values.
    """

    errors: list[str] = []

    for column_name, definition in schema["columns"].items():

        allowed_values = definition.get(
            "allowed_values"
        )

        if allowed_values is None:
            continue

        observed_values = set(
            df[column_name].dropna().unique()
        )

        invalid_values = (
            observed_values
            - set(allowed_values)
        )

        if invalid_values:
            errors.append(
                f"{column_name}: invalid values "
                f"{sorted(invalid_values)}"
            )

    if errors:
        raise SchemaError(
            "Allowed-value validation failed:\n"
            + "\n".join(errors)
        )


def validate_numeric_constraints(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> None:
    """
    Validate numeric minimum constraints defined by the schema.
    """

    errors: list[str] = []

    for column_name, definition in schema["columns"].items():

        constraints = definition.get(
            "constraints",
            {},
        )

        minimum = constraints.get("min")

        if minimum is not None:

            values = df[column_name]

            invalid_count = int(
                (values < minimum).sum()
            )

            if invalid_count > 0:
                errors.append(
                    f"{column_name}: "
                    f"{invalid_count:,} values "
                    f"below minimum {minimum}"
                )

        minimum_length = constraints.get(
            "min_length"
        )

        if minimum_length is not None:

            values = (
                df[column_name]
                .astype("string")
            )

            invalid_count = int(
                (
                    values.str.len()
                    < minimum_length
                ).sum()
            )

            if invalid_count > 0:
                errors.append(
                    f"{column_name}: "
                    f"{invalid_count:,} values "
                    f"shorter than {minimum_length}"
                )

    if errors:
        raise SchemaError(
            "Numeric/string constraint validation failed:\n"
            + "\n".join(errors)
        )


def validate_schema(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> None:
    """
    Execute the complete schema contract against a DataFrame.

    This function does not modify the DataFrame.
    """

    validate_schema_definition(schema)
    validate_columns(df, schema)
    validate_column_order(df, schema)
    validate_nullability(df, schema)
    validate_allowed_values(df, schema)
    validate_numeric_constraints(df, schema)