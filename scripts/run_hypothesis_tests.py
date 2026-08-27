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

from src.data.hypothesis_testing import (
    run_continuous_hypotheses,
    run_transaction_type_hypothesis,
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
    / "hypothesis_testing"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — HYPOTHESIS TESTING"
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
        "CONTINUOUS HYPOTHESES"
    )

    print(
        "Test: two-sided Mann-Whitney U"
    )

    print(
        "Alpha: 0.05"
    )

    continuous_results = (
        run_continuous_hypotheses(df)
    )

    print(
        continuous_results.to_string(
            index=False
        )
    )

    print_header(
        "TRANSACTION TYPE HYPOTHESIS"
    )

    print(
        "H0: Transaction type and fraud "
        "status are independent."
    )

    print(
        "H1: Transaction type and fraud "
        "status are associated."
    )

    contingency, categorical_result = (
        run_transaction_type_hypothesis(
            df
        )
    )

    print(
        "\nObserved contingency table:"
    )

    print(contingency)

    print(
        "\nChi-square result:"
    )

    print(
        categorical_result.to_string(
            index=False
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    continuous_path = (
        REPORT_DIR
        / "continuous_hypotheses.csv"
    )

    contingency_path = (
        REPORT_DIR
        / "transaction_type_contingency.csv"
    )

    categorical_path = (
        REPORT_DIR
        / "transaction_type_hypothesis.csv"
    )

    continuous_results.to_csv(
        continuous_path,
        index=False,
    )

    contingency.to_csv(
        contingency_path
    )

    categorical_result.to_csv(
        categorical_path,
        index=False,
    )

    print_header(
        "HYPOTHESIS TESTING COMPLETE"
    )

    print(
        f"Saved: {continuous_path}"
    )

    print(
        f"Saved: {contingency_path}"
    )

    print(
        f"Saved: {categorical_path}"
    )

    print(
        "\nCanonical dataset was not modified."
    )

    print(
        "No statistical transformations were applied."
    )


if __name__ == "__main__":
    main()