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
)

from src.data.quality_findings import (
    QualityFindingsEngine,
    QualityFindingsError,
    findings_to_dataframe,
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

OUTPUT_PATH = (
    OUTPUT_DIR
    / "quality_findings.csv"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — AUTOMATED QUALITY FINDINGS"
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

    # --------------------------------------------------------------
    # Generate quality report in memory
    # --------------------------------------------------------------

    print_header(
        "BUILDING QUALITY METRICS"
    )

    try:

        report_generator = (
            DataQualityReport(df)
        )

        report = (
            report_generator.generate()
        )

    except QualityReportError as exc:

        print(
            "\nQUALITY REPORT FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    # --------------------------------------------------------------
    # Generate findings
    # --------------------------------------------------------------

    print_header(
        "GENERATING AUTOMATED FINDINGS"
    )

    try:

        engine = QualityFindingsEngine(
            report
        )

        findings = engine.generate()

    except QualityFindingsError as exc:

        print(
            "\nQUALITY FINDINGS FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    findings_df = (
        findings_to_dataframe(
            findings
        )
    )

    # --------------------------------------------------------------
    # Print findings
    # --------------------------------------------------------------

    print()

    for finding in findings:

        print(
            f"[{finding.severity:<8}] "
            f"{finding.finding_id} | "
            f"{finding.category:<18} | "
            f"{finding.finding}"
        )

        print(
            f"           Metric: "
            f"{finding.metric} = "
            f"{finding.value}"
        )

        print(
            f"           Implication: "
            f"{finding.downstream_implication}"
        )

        print()

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print_header(
        "FINDINGS SUMMARY"
    )

    severity_counts = (
        findings_df[
            "severity"
        ]
        .value_counts()
    )

    for severity in [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "INFO",
    ]:

        print(
            f"{severity:<10}: "
            f"{int(severity_counts.get(severity, 0))}"
        )

    # --------------------------------------------------------------
    # Persist
    # --------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    findings_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_header(
        "AUTOMATED QUALITY FINDINGS COMPLETE"
    )

    print(
        f"Findings generated: "
        f"{len(findings):,}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\nNo data was modified."
    )


if __name__ == "__main__":
    main()