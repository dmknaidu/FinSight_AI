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

from src.data.quality_report import (
    DataQualityReport,
    QualityReportError,
    save_quality_report,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_engineering"
    / "quality"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — DATA QUALITY REPORT"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

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

    print_header(
        "GENERATING QUALITY ANALYSIS"
    )

    try:

        analyzer = DataQualityReport(
            df
        )

        report = analyzer.generate()

    except QualityReportError as exc:

        print(
            "\nQUALITY REPORT FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print_header(
        "QUALITY REPORT SUMMARY"
    )

    summary = report[
        "dataset_summary"
    ]

    for _, row in summary.iterrows():

        print(
            f"{row['metric']:<35}: "
            f"{row['value']}"
        )

    # --------------------------------------------------------------
    # Fraud summary
    # --------------------------------------------------------------

    fraud = report[
        "fraud_profile"
    ]

    if not fraud.empty:

        print_header(
            "FRAUD PROFILE"
        )

        for _, row in fraud.iterrows():

            print(
                f"{row['metric']:<35}: "
                f"{row['value']}"
            )

    # --------------------------------------------------------------
    # Transaction types
    # --------------------------------------------------------------

    transaction_types = report[
        "transaction_types"
    ]

    if not transaction_types.empty:

        print_header(
            "TRANSACTION TYPE PROFILE"
        )

        print(
            transaction_types.to_string(
                index=False
            )
        )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_quality_report(
        report,
        OUTPUT_DIR,
    )

    print_header(
        "QUALITY REPORT COMPLETE"
    )

    print(
        f"Reports saved under:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nNo data was modified."
    )


if __name__ == "__main__":
    main()