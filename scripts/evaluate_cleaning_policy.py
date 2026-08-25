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

from src.data.cleaning_evaluator import (
    CleaningPolicyEvaluator,
    build_policy_report,
    calculate_policy_summary,
)

from src.data.cleaning_policy import (
    Treatment,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_engineering"
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
        "FINSIGHT AI — CLEANING POLICY EVALUATION"
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
    # Evaluate policy
    # ------------------------------------------------------------------

    evaluator = CleaningPolicyEvaluator(
        expected_columns=EXPECTED_COLUMNS
    )

    print_header(
        "EVALUATING CLEANING POLICY"
    )

    evaluations = evaluator.evaluate(
        df
    )

    for evaluation in evaluations:

        print(
            f"{evaluation.issue:<35}"
            f" | "
            f"{evaluation.treatment:<18}"
            f" | "
            f"{evaluation.count:>12,}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = calculate_policy_summary(
        evaluations
    )

    print_header(
        "POLICY IMPACT SUMMARY"
    )

    print(
        f"Pipeline failures : "
        f"{summary[Treatment.PIPELINE_FAILURE.value]:,}"
    )

    print(
        f"Quarantine        : "
        f"{summary[Treatment.QUARANTINE.value]:,}"
    )

    print(
        f"Preserve + flag   : "
        f"{summary[Treatment.PRESERVE_FLAG.value]:,}"
    )

    print(
        f"Preserve           : "
        f"{summary[Treatment.PRESERVE.value]:,}"
    )

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIR
        / "cleaning_policy_evaluation.csv"
    )

    report = build_policy_report(
        evaluations
    )

    report.to_csv(
        report_path,
        index=False,
    )

    print(
        f"\nSaved: {report_path}"
    )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    print_header(
        "CLEANING POLICY DECISION"
    )

    pipeline_failures = summary[
        Treatment.PIPELINE_FAILURE.value
    ]

    quarantine = summary[
        Treatment.QUARANTINE.value
    ]

    if pipeline_failures > 0:

        print(
            "PIPELINE FAILURE CONDITIONS DETECTED."
        )

        print(
            "The dataset requires structural investigation."
        )

        raise SystemExit(1)

    if quarantine > 0:

        print(
            "QUARANTINE CANDIDATES DETECTED."
        )

        print(
            "No records were modified or removed."
        )

        print(
            "The next cleaning step will determine "
            "how quarantine records are persisted."
        )

    else:

        print(
            "No records currently require quarantine."
        )

        print(
            "The dataset can proceed without "
            "destructive cleaning."
        )

    print(
        "\nPolicy evaluation completed."
    )

    print(
        "No data was modified."
    )


if __name__ == "__main__":
    main()