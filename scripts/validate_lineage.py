from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.ingestion import (
    DataIngestionError,
    ingest_csv,
)

from src.data.lineage_validation import (
    LineageValidationError,
    load_manifest,
    validate_lineage,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_engineering"
    / "lineage"
    / "lineage_manifest.json"
)


def print_header(
    title: str,
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — LINEAGE VALIDATION"
    )

    print(
        f"Dataset : {DATASET_PATH}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    # --------------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------------

    print(
        "\nLoading lineage manifest..."
    )

    try:

        manifest = load_manifest(
            MANIFEST_PATH
        )

    except LineageValidationError as exc:

        print(
            "\nMANIFEST LOAD FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print(
        "Manifest loaded successfully."
    )

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    try:

        df, metadata = ingest_csv(
            DATASET_PATH
        )

    except DataIngestionError as exc:

        print(
            "\nDATA INGESTION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print(
        f"Rows: {metadata.rows:,}"
    )

    print(
        f"Columns: {metadata.columns}"
    )

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    print_header(
        "RUNNING LINEAGE VALIDATION"
    )

    try:

        results = validate_lineage(
            dataset_path=DATASET_PATH,
            df=df,
            manifest=manifest,
        )

    except LineageValidationError as exc:

        print(
            "\nLINEAGE VALIDATION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

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
        "LINEAGE VALIDATION SUMMARY"
    )

    print(
        f"PASS : {passed}"
    )

    print(
        f"FAIL : {failed}"
    )

    if failed > 0:

        print_header(
            "LINEAGE VALIDATION FAILED"
        )

        print(
            "The dataset does not match the "
            "recorded lineage manifest."
        )

        raise SystemExit(1)

    print_header(
        "LINEAGE VALIDATION COMPLETE"
    )

    print(
        "All lineage and reproducibility "
        "checks passed."
    )

    print(
        "\nNo data was modified."
    )


if __name__ == "__main__":
    main()