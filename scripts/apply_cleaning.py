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

from src.data.cleaning import (
    CleaningError,
    apply_cleaning_policy,
    validate_cleaning_result,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — CLEANING PIPELINE"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    # ------------------------------------------------------------------
    # Load
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
    # Apply cleaning policy
    # ------------------------------------------------------------------

    print_header(
        "APPLYING CLEANING POLICY"
    )

    try:

        result = apply_cleaning_policy(
            df
        )

    except CleaningError as exc:

        print("\nCLEANING FAILED")
        print("-" * 100)
        print(str(exc))

        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Validate result
    # ------------------------------------------------------------------

    print_header(
        "VALIDATING CLEANING RESULT"
    )

    try:

        validate_cleaning_result(
            df,
            result,
        )

    except CleaningError as exc:

        print("\nCLEANING VALIDATION FAILED")
        print("-" * 100)
        print(str(exc))

        raise SystemExit(1)

    print(
        "[PASS] Canonical + quarantine rows "
        "equal original row count"
    )

    print(
        "[PASS] Canonical and quarantine partitions "
        "are mutually exclusive"
    )

    print(
        "[PASS] Every original row is accounted for"
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    print_header(
        "CLEANING RESULTS"
    )

    print(
        f"Original rows       : "
        f"{result.total_rows:,}"
    )

    print(
        f"Canonical rows      : "
        f"{result.canonical_rows:,}"
    )

    print(
        f"Quarantine rows     : "
        f"{result.quarantine_rows:,}"
    )

    if result.total_rows:

        quarantine_rate = (
            result.quarantine_rows
            / result.total_rows
            * 100
        )

    else:

        quarantine_rate = 0.0

    print(
        f"Quarantine rate     : "
        f"{quarantine_rate:.6f}%"
    )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    print_header(
        "CLEANING DECISION"
    )

    if result.quarantine_rows == 0:

        print(
            "No records required quarantine."
        )

        print(
            "The canonical dataset contains "
            "all source records."
        )

    else:

        print(
            f"{result.quarantine_rows:,} records "
            "were isolated for quarantine."
        )

        print(
            "Quarantined records were not deleted."
        )

    print(
        "\nNo source records were permanently deleted."
    )

    print(
        "No output dataset has been persisted yet."
    )

    print(
        "Persistence will be handled by the "
        "canonical dataset step."
    )


if __name__ == "__main__":
    main()