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

from src.data.distribution_analysis import (
    DistributionAnalysisConfig,
    calculate_distribution_profile,
    calculate_normality_profile,
    interpret_distribution,
    merge_distribution_results,
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
    / "distributions"
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
        "FINSIGHT AI — DISTRIBUTION & NORMALITY ANALYSIS"
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

    print_header(
        "CALCULATING DISTRIBUTION PROFILES"
    )

    distribution_profile = (
        calculate_distribution_profile(df)
    )

    print(
        distribution_profile.to_string(
            index=False
        )
    )

    print_header(
        "NORMALITY ANALYSIS"
    )

    config = (
        DistributionAnalysisConfig()
    )

    print(
        f"Test           : Shapiro-Wilk"
    )

    print(
        f"Sample size    : {config.sample_size:,}"
    )

    print(
        f"Random seed    : {config.random_seed}"
    )

    normality_profile = (
        calculate_normality_profile(
            df,
            config,
        )
    )

    print(
        normality_profile.to_string(
            index=False
        )
    )

    print_header(
        "COMBINED STATISTICAL ASSESSMENT"
    )

    combined = merge_distribution_results(
        distribution_profile,
        normality_profile,
    )

    interpreted = interpret_distribution(
        combined
    )

    display_columns = [
        "column",
        "skewness",
        "kurtosis",
        "zero_percentage",
        "iqr_outlier_percentage",
        "normality_assessment",
        "distribution_assessment",
        "recommended_statistical_family",
    ]

    print(
        interpreted[
            display_columns
        ].to_string(index=False)
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution_path = (
        REPORT_DIR
        / "distribution_profile.csv"
    )

    normality_path = (
        REPORT_DIR
        / "normality_profile.csv"
    )

    assessment_path = (
        REPORT_DIR
        / "distribution_assessment.csv"
    )

    distribution_profile.to_csv(
        distribution_path,
        index=False,
    )

    normality_profile.to_csv(
        normality_path,
        index=False,
    )

    interpreted.to_csv(
        assessment_path,
        index=False,
    )

    print_header(
        "DISTRIBUTION ANALYSIS COMPLETE"
    )

    print(
        f"Saved: {distribution_path}"
    )

    print(
        f"Saved: {normality_path}"
    )

    print(
        f"Saved: {assessment_path}"
    )

    print(
        "\nCanonical dataset was not modified."
    )

    print(
        "No statistical transformations were applied."
    )


if __name__ == "__main__":
    main()