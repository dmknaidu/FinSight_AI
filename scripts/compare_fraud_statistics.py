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


import pandas as pd

from src.data.fraud_statistical_comparison import (
    calculate_group_descriptives,
    calculate_mann_whitney_results,
    apply_fdr_correction,
    create_comparison_summary,
)


CANONICAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "finsight_canonical.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "statistical"
    / "fraud_comparison"
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
        "FINSIGHT AI — FRAUD/NON-FRAUD STATISTICAL COMPARISON"
    )

    print(
        f"Canonical dataset:\n"
        f"{CANONICAL_PATH}"
    )

    print(
        "\nLoading canonical dataset..."
    )

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    fraud_count = int(
        df["isFraud"].sum()
    )

    legitimate_count = (
        len(df) - fraud_count
    )

    fraud_rate = (
        fraud_count / len(df) * 100
    )

    print_header(
        "COMPARISON POPULATION"
    )

    print(
        f"Legitimate transactions : "
        f"{legitimate_count:,}"
    )

    print(
        f"Fraud transactions      : "
        f"{fraud_count:,}"
    )

    print(
        f"Fraud rate              : "
        f"{fraud_rate:.6f}%"
    )

    print_header(
        "GROUP DESCRIPTIVE STATISTICS"
    )

    descriptives = (
        calculate_group_descriptives(
            df
        )
    )

    print(
        descriptives.to_string(
            index=False
        )
    )

    print_header(
        "MANN-WHITNEY U TESTS"
    )

    print(
        "Alternative : two-sided"
    )

    print(
        "Significance level : 0.05"
    )

    results = (
        calculate_mann_whitney_results(
            df
        )
    )

    results = apply_fdr_correction(
        results,
        alpha=0.05,
    )

    print(
        results.to_string(
            index=False
        )
    )

    print_header(
        "STATISTICAL COMPARISON SUMMARY"
    )

    summary = (
        create_comparison_summary(
            results
        )
    )

    display_columns = [
        "column",
        "p_value",
        "adjusted_p_value",
        "significant_after_fdr",
        "rank_biserial_correlation",
        "effect_direction",
        "legitimate_median",
        "fraud_median",
        "median_difference",
        "statistical_conclusion",
    ]

    print(
        summary[
            display_columns
        ].to_string(index=False)
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptives_path = (
        REPORT_DIR
        / "group_descriptives.csv"
    )

    results_path = (
        REPORT_DIR
        / "mann_whitney_results.csv"
    )

    summary_path = (
        REPORT_DIR
        / "statistical_comparison_summary.csv"
    )

    descriptives.to_csv(
        descriptives_path,
        index=False,
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print_header(
        "STATISTICAL COMPARISON COMPLETE"
    )

    print(
        f"Saved: {descriptives_path}"
    )

    print(
        f"Saved: {results_path}"
    )

    print(
        f"Saved: {summary_path}"
    )

    print(
        "\nCanonical dataset was not modified."
    )

    print(
        "No statistical transformations were applied."
    )


if __name__ == "__main__":
    main()