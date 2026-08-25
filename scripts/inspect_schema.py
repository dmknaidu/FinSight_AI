from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.ingestion import (
    DataIngestionError,
    ingest_csv,
)

from src.data.schema import (
    SchemaError,
    load_schema,
    validate_schema,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "configs"
    / "schema.yaml"
)


def print_header(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def main() -> None:

    print_header(
        "FINSIGHT AI — SCHEMA CONTRACT"
    )

    print(
        f"Schema: {SCHEMA_PATH}"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    # ------------------------------------------------------------------
    # Load schema
    # ------------------------------------------------------------------

    print("\nLoading schema...")

    try:
        schema = load_schema(
            SCHEMA_PATH
        )

    except SchemaError as exc:
        print("\nSCHEMA LOAD FAILED")
        print("-" * 90)
        print(str(exc))
        raise SystemExit(1)

    print(
        "Schema loaded successfully."
    )

    # ------------------------------------------------------------------
    # Display schema
    # ------------------------------------------------------------------

    print_header(
        "SCHEMA DEFINITION"
    )

    for column_name, definition in (
        schema["columns"].items()
    ):

        print(
            f"{column_name:<18}"
            f"dtype={definition['dtype']:<10}"
            f"nullable={str(definition['nullable']):<6}"
        )

        description = definition.get(
            "description"
        )

        if description:
            print(
                f"  Description: {description.strip()}"
            )

        allowed_values = definition.get(
            "allowed_values"
        )

        if allowed_values:
            print(
                f"  Allowed values: {allowed_values}"
            )

        constraints = definition.get(
            "constraints"
        )

        if constraints:
            print(
                f"  Constraints: {constraints}"
            )

    # ------------------------------------------------------------------
    # Ingest dataset
    # ------------------------------------------------------------------

    print_header(
        "LOADING DATASET"
    )

    try:
        df, metadata = ingest_csv(
            DATASET_PATH
        )

    except DataIngestionError as exc:
        print("\nDATA INGESTION FAILED")
        print("-" * 90)
        print(str(exc))
        raise SystemExit(1)

    print(
        f"Rows: {metadata.rows:,}"
    )

    print(
        f"Columns: {metadata.columns}"
    )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    print_header(
        "SCHEMA VALIDATION"
    )

    try:
        validate_schema(
            df,
            schema,
        )

    except SchemaError as exc:
        print("\nSCHEMA VALIDATION FAILED")
        print("-" * 90)
        print(str(exc))
        raise SystemExit(1)

    print(
        "[PASS] Schema definition"
    )

    print(
        "[PASS] Required columns"
    )

    print(
        "[PASS] Column order"
    )

    print(
        "[PASS] Nullability"
    )

    print(
        "[PASS] Allowed values"
    )

    print(
        "[PASS] Numeric/string constraints"
    )

    # ------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------

    print_header(
        "SCHEMA CONTRACT COMPLETE"
    )

    print(
        "Dataset conforms to the Phase 1 schema contract."
    )

    print(
        "\nNo data was modified."
    )


if __name__ == "__main__":
    main()