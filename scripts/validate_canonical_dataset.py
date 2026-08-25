from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.data.ingestion import (
    DataIngestionError,
    ingest_csv,
)

from src.data.canonical_validation import (
    validate_canonical_dataset,
)


SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

CANONICAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "finsight_canonical.parquet"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — CANONICAL DATASET VALIDATION"
    )

    print(
        f"Source    : {SOURCE_PATH}"
    )

    print(
        f"Canonical : {CANONICAL_PATH}"
    )

    # --------------------------------------------------------------
    # Load source
    # --------------------------------------------------------------

    print(
        "\nLoading source dataset..."
    )

    try:

        source_df, metadata = ingest_csv(
            SOURCE_PATH
        )

    except DataIngestionError as exc:

        print(
            "\nSOURCE INGESTION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print(
        f"Source rows    : {metadata.rows:,}"
    )

    print(
        f"Source columns : {metadata.columns}"
    )

    # --------------------------------------------------------------
    # Load canonical
    # --------------------------------------------------------------

    print(
        "\nLoading canonical Parquet..."
    )

    try:

        canonical_df = pd.read_parquet(
            CANONICAL_PATH
        )

    except Exception as exc:

        print(
            "\nCANONICAL DATASET LOAD FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print(
        f"Canonical rows    : {len(canonical_df):,}"
    )

    print(
        f"Canonical columns : {len(canonical_df.columns)}"
    )

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    print_header(
        "RUNNING CANONICAL VALIDATION"
    )

    results = validate_canonical_dataset(
        source_df,
        canonical_df,
    )

    for result in results:

        print(
            f"[{result.status}] "
            f"{result.name:<30} "
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
        "CANONICAL VALIDATION SUMMARY"
    )

    print(
        f"PASS : {passed}"
    )

    print(
        f"FAIL : {failed}"
    )

    if failed:

        print_header(
            "CANONICAL VALIDATION FAILED"
        )

        print(
            "The persisted canonical dataset "
            "does not satisfy its contract."
        )

        raise SystemExit(1)

    print_header(
        "CANONICAL DATASET VALIDATION COMPLETE"
    )

    print(
        "All canonical dataset integrity checks passed."
    )

    print(
        "\nSource data was not modified."
    )


if __name__ == "__main__":
    main()