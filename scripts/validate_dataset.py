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

from src.data.validation_pipeline import (
    DataValidator,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)


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


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — DATA VALIDATION PIPELINE"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    print("\nLoading dataset...")

    try:

        df, metadata = ingest_csv(
            DATASET_PATH
        )

    except DataIngestionError as exc:

        print("\nDATA INGESTION FAILED")
        print("-" * 100)
        print(str(exc))

        raise SystemExit(1)

    print(
        f"Rows: {metadata.rows:,}"
    )

    print(
        f"Columns: {metadata.columns}"
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    validator = DataValidator(
        expected_columns=EXPECTED_COLUMNS
    )

    print_header(
        "RUNNING VALIDATION CHECKS"
    )

    results = validator.validate(
        df
    )

    for result in results:

        print(
            f"[{result.status}] "
            f"{result.name:<35} "
            f"{result.message}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    passed = sum(
        result.status == "PASS"
        for result in results
    )

    warnings = sum(
        result.status == "WARNING"
        for result in results
    )

    failed = sum(
        result.status == "FAIL"
        for result in results
    )

    print_header(
        "VALIDATION SUMMARY"
    )

    print(
        f"PASS     : {passed}"
    )

    print(
        f"WARNING  : {warnings}"
    )

    print(
        f"FAIL     : {failed}"
    )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    if failed > 0:

        print_header(
            "VALIDATION FAILED"
        )

        print(
            "The dataset contains validation failures."
        )

        print(
            "No data was modified."
        )

        raise SystemExit(1)

    print_header(
        "VALIDATION COMPLETE"
    )

    if warnings > 0:

        print(
            "Dataset passed all blocking checks "
            "but contains warnings requiring review."
        )

    else:

        print(
            "Dataset passed all validation checks."
        )

    print(
        "No data was modified."
    )


if __name__ == "__main__":
    main()