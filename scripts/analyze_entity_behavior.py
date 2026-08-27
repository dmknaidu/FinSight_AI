#!/usr/bin/env python3

"""
FinSight AI — Entity-Level Behavioral Analysis

Phase 3 — Step 8.

Loads the immutable canonical dataset, performs entity-level
behavioral/statistical analysis, and writes analytical reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.entity_behavior_analysis import (
    analyze_entity_behavior,
    save_entity_analysis,
)


CANONICAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "finsight_canonical.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "statistical"
    / "entity_behavior"
)


def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_dataframe(
    dataframe: pd.DataFrame,
    max_rows: int = 20,
) -> None:
    if dataframe.empty:
        print("(no rows)")
        return

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_rows",
        max_rows,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(dataframe.head(max_rows).to_string(index=False))


def main() -> None:
    print("=" * 100)
    print("FINSIGHT AI — ENTITY-LEVEL BEHAVIORAL ANALYSIS")
    print("=" * 100)
    print(f"Canonical dataset:\n{CANONICAL_DATASET}")

    if not CANONICAL_DATASET.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {CANONICAL_DATASET}"
        )

    print()
    print("Loading canonical dataset...")

    df = pd.read_parquet(CANONICAL_DATASET)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print_section("ENTITY ANALYSIS POPULATION")

    fraud_count = int(df["isFraud"].sum())
    total_count = len(df)

    print(f"Total transactions : {total_count:,}")
    print(f"Fraud transactions : {fraud_count:,}")
    print(
        f"Fraud rate         : "
        f"{fraud_count / total_count * 100:.6f}%"
    )
    print(
        f"Origin entities    : "
        f"{df['nameOrig'].nunique():,}"
    )
    print(
        f"Destination entities: "
        f"{df['nameDest'].nunique():,}"
    )

    print_section("BUILDING ENTITY PROFILES")

    results = analyze_entity_behavior(df)

    origin_profile = results["origin_entity_profile"]
    destination_profile = results["destination_entity_profile"]

    print(
        f"Origin profile entities      : "
        f"{len(origin_profile):,}"
    )
    print(
        f"Destination profile entities : "
        f"{len(destination_profile):,}"
    )

    print_section("ORIGIN ENTITY PROFILE")
    print_dataframe(
        origin_profile.sort_values(
            "transaction_count",
            ascending=False,
        ),
        max_rows=15,
    )

    print_section("DESTINATION ENTITY PROFILE")
    print_dataframe(
        destination_profile.sort_values(
            "transaction_count",
            ascending=False,
        ),
        max_rows=15,
    )

    print_section("ORIGIN REUSE DISTRIBUTION")
    print_dataframe(
        results["origin_reuse_distribution"],
        max_rows=20,
    )

    print_section("DESTINATION REUSE DISTRIBUTION")
    print_dataframe(
        results["destination_reuse_distribution"],
        max_rows=20,
    )

    print_section("ORIGIN FRAUD ASSOCIATION")
    print_dataframe(
        results["origin_fraud_association"],
        max_rows=20,
    )

    print_section("DESTINATION FRAUD ASSOCIATION")
    print_dataframe(
        results["destination_fraud_association"],
        max_rows=20,
    )

    print_section("ENTITY FRAUD CONCENTRATION")
    print_dataframe(
        results["entity_fraud_concentration"],
        max_rows=20,
    )

    print_section("ENTITY BEHAVIOR COMPARISON")
    print_dataframe(
        results["entity_behavior_summary"],
        max_rows=30,
    )

    print_section("SAVING ENTITY ANALYSIS")

    save_entity_analysis(
        results,
        OUTPUT_DIRECTORY,
    )

    for filename in results:
        print(
            "Saved: "
            f"{OUTPUT_DIRECTORY / (filename + '.csv')}"
        )

    print()
    print("=" * 100)
    print("ENTITY-LEVEL BEHAVIORAL ANALYSIS COMPLETE")
    print("=" * 100)
    print()
    print("Canonical dataset was not modified.")
    print("No ML features were created.")
    print("No canonical statistical transformations were applied.")


if __name__ == "__main__":
    main()