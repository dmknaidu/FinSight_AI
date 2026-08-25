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
)

from src.data.cleaning_validation import (
    validate_cleaning,
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
        "FINSIGHT AI — CLEANING VALIDATION"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    print("\nLoading dataset...")

    try:

        original, metadata = ingest_csv(
            DATASET_PATH
        )

    except DataIngestionError as exc:

        print("\nDATA INGESTION FAILED")
        print(str(exc))

        raise SystemExit(1)

    print(
        f"Rows: {metadata.rows:,}"
    )

    print(
        f"Columns: {metadata.columns}"
    )

    # --------------------------------------------------------------
    # Apply cleaning in memory
    # --------------------------------------------------------------

    print_header(
        "REBUILDING CLEANING PARTITIONS"
    )

    try:

        cleaning_result = (
            apply_cleaning_policy(
                original
            )
        )

    except CleaningError as exc:

        print("\nCLEANING FAILED")
        print(str(exc))

        raise SystemExit(1)

    canonical = (
        cleaning_result.canonical
    )

    quarantine = (
        cleaning_result.quarantine
    )

    print(
        f"Canonical rows  : "
        f"{len(canonical):,}"
    )

    print(
        f"Quarantine rows : "
        f"{len(quarantine):,}"
    )

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    print_header(
        "RUNNING CLEANING VALIDATION"
    )

    results = validate_cleaning(
        original,
        canonical,
        quarantine,
    )

    for result in results:

        print(
            f"[{result.status}] "
            f"{result.name:<35} "
            f"{result.message}"
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    passed = sum(
        result.status == "PASS"
        for result in results
    )

    failed = sum(
        result.status == "FAIL"
        for result in results
    )

    print_header(
        "CLEANING VALIDATION SUMMARY"
    )

    print(
        f"PASS : {passed}"
    )

    print(
        f"FAIL : {failed}"
    )

    if failed > 0:

        print_header(
            "CLEANING VALIDATION FAILED"
        )

        print(
            "The cleaning implementation did not "
            "satisfy all integrity checks."
        )

        raise SystemExit(1)

    print_header(
        "CLEANING VALIDATION COMPLETE"
    )

    print(
        "All cleaning integrity checks passed."
    )

    print(
        "No source data was modified."
    )


if __name__ == "__main__":
    main()