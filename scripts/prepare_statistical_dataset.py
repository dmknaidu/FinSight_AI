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

from src.data.statistical_preparation import (
    StatisticalPreparationError,
    calculate_continuous_statistics,
    calculate_type_profile,
    prepare_statistical_dataset,
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
        "FINSIGHT AI — STATISTICAL DATASET PREPARATION"
    )

    print(
        f"Canonical dataset:\n"
        f"{CANONICAL_PATH}"
    )

    print(
        "\nLoading canonical dataset..."
    )

    try:

        df = pd.read_parquet(
            CANONICAL_PATH
        )

        print(
            f"Rows    : {len(df):,}"
        )

        print(
            f"Columns : {len(df.columns)}"
        )

        prepared_df, summary = (
            prepare_statistical_dataset(df)
        )

    except (
        StatisticalPreparationError,
        FileNotFoundError,
    ) as exc:

        print(
            "\nSTATISTICAL PREPARATION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print_header(
        "STATISTICAL POPULATION"
    )

    print(
        f"Total transactions : "
        f"{summary.rows:,}"
    )

    print(
        f"Fraud transactions : "
        f"{summary.fraud_transactions:,}"
    )

    print(
        f"Legitimate         : "
        f"{summary.legitimate_transactions:,}"
    )

    print(
        f"Fraud rate         : "
        f"{summary.fraud_rate * 100:.6f}%"
    )

    print(
        f"Missing values     : "
        f"{summary.missing_values:,}"
    )

    print(
        f"Infinite values    : "
        f"{summary.infinite_values:,}"
    )

    print_header(
        "VARIABLE CLASSIFICATION"
    )

    print(
        "Continuous : "
        "amount, oldbalanceOrg, newbalanceOrig, "
        "oldbalanceDest, newbalanceDest"
    )

    print(
        "Temporal   : step"
    )

    print(
        "Categorical: type"
    )

    print(
        "Entities   : nameOrig, nameDest"
    )

    print(
        "Binary     : isFraud, isFlaggedFraud"
    )

    print(
        "Target     : isFraud"
    )

    print_header(
        "CONTINUOUS VARIABLE STATISTICS"
    )

    continuous_stats = (
        calculate_continuous_statistics(
            prepared_df
        )
    )

    print(
        continuous_stats.to_string(
            index=False
        )
    )

    print_header(
        "TRANSACTION TYPE PROFILE"
    )

    type_profile = (
        calculate_type_profile(
            prepared_df
        )
    )

    print(
        type_profile.to_string(
            index=False
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    continuous_path = (
        REPORT_DIR
        / "continuous_statistics.csv"
    )

    type_path = (
        REPORT_DIR
        / "transaction_type_profile.csv"
    )

    continuous_stats.to_csv(
        continuous_path,
        index=False,
    )

    type_profile.to_csv(
        type_path,
        index=False,
    )

    print_header(
        "STATISTICAL PREPARATION COMPLETE"
    )

    print(
        f"Saved: {continuous_path}"
    )

    print(
        f"Saved: {type_path}"
    )

    print(
        "\nCanonical dataset was not modified."
    )

    print(
        "No statistical transformations were applied."
    )


if __name__ == "__main__":
    main()