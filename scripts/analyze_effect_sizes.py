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

from src.data.effect_size_analysis import (
    calculate_effect_sizes,
    rank_effect_sizes,
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
    / "effect_size"
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
        "FINSIGHT AI — EFFECT SIZE ANALYSIS"
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
        "CALCULATING EFFECT SIZES"
    )

    effect_profile = (
        calculate_effect_sizes(df)
    )

    print(
        effect_profile.to_string(
            index=False
        )
    )

    print_header(
        "EFFECT SIZE RANKING"
    )

    ranked = rank_effect_sizes(
        effect_profile
    )

    display_columns = [
        "effect_rank",
        "column",
        "rank_biserial_correlation",
        "absolute_effect_size",
        "effect_magnitude",
        "effect_direction",
        "probability_fraud_greater",
        "legitimate_median",
        "fraud_median",
        "median_difference",
        "median_ratio",
    ]

    print(
        ranked[
            display_columns
        ].to_string(
            index=False
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_path = (
        REPORT_DIR
        / "effect_size_profile.csv"
    )

    ranking_path = (
        REPORT_DIR
        / "effect_size_ranking.csv"
    )

    effect_profile.to_csv(
        profile_path,
        index=False,
    )

    ranked.to_csv(
        ranking_path,
        index=False,
    )

    print_header(
        "EFFECT SIZE ANALYSIS COMPLETE"
    )

    print(
        f"Saved: {profile_path}"
    )

    print(
        f"Saved: {ranking_path}"
    )

    print(
        "\nCanonical dataset was not modified."
    )

    print(
        "No statistical transformations were applied."
    )


if __name__ == "__main__":
    main()