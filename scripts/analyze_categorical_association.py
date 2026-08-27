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

from src.data.categorical_association import (
    run_association_analysis,
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
    / "categorical_association"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — CATEGORICAL ASSOCIATION ANALYSIS"
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
        "CALCULATING CATEGORICAL ASSOCIATION"
    )

    results = run_association_analysis(
        df
    )

    print_header(
        "CATEGORY PROFILE"
    )

    print(
        results["profile"].to_string(
            index=False
        )
    )

    print_header(
        "OBSERVED CONTINGENCY TABLE"
    )

    print(
        results["observed"].to_string()
    )

    print_header(
        "EXPECTED CONTINGENCY TABLE"
    )

    print(
        results["expected"].to_string()
    )

    print_header(
        "STANDARDIZED PEARSON RESIDUALS"
    )

    print(
        results["residuals"].to_string()
    )

    print_header(
        "CHI-SQUARE CONTRIBUTIONS"
    )

    print(
        results["contributions"].to_string(
            index=False
        )
    )

    print_header(
        "ASSOCIATION SUMMARY"
    )

    print(
        results["summary"].to_string(
            index=False
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "profile": (
            REPORT_DIR
            / "categorical_profile.csv"
        ),
        "observed": (
            REPORT_DIR
            / "observed_contingency.csv"
        ),
        "expected": (
            REPORT_DIR
            / "expected_contingency.csv"
        ),
        "residuals": (
            REPORT_DIR
            / "standardized_residuals.csv"
        ),
        "contributions": (
            REPORT_DIR
            / "chi_square_contributions.csv"
        ),
        "summary": (
            REPORT_DIR
            / "categorical_association_summary.csv"
        ),
    }

    results["profile"].to_csv(
        paths["profile"],
        index=False,
    )

    results["observed"].to_csv(
        paths["observed"]
    )

    results["expected"].to_csv(
        paths["expected"]
    )

    results["residuals"].to_csv(
        paths["residuals"]
    )

    results["contributions"].to_csv(
        paths["contributions"],
        index=False,
    )

    results["summary"].to_csv(
        paths["summary"],
        index=False,
    )

    print_header(
        "CATEGORICAL ASSOCIATION ANALYSIS COMPLETE"
    )

    for path in paths.values():
        print(f"Saved: {path}")

    print(
        "\nCanonical dataset was not modified."
    )

    print(
        "No statistical transformations were applied."
    )


if __name__ == "__main__":
    main()