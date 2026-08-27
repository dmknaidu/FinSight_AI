"""
FinSight AI — Relationship & Network-Level Fraud Analysis Script
=================================================================

Runs Step 9 relationship and network-level fraud analysis against the
canonical dataset.

The canonical dataset is read only and is never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# Project path configuration
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.data.relationship_analysis import (  # noqa: E402
    RelationshipAnalysis,
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
    / "relationship_analysis"
)


# ----------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------

def print_header(title: str) -> None:
    """Print a standardized analysis section header."""

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


def print_dataframe(
    dataframe: pd.DataFrame,
    max_rows: int = 15,
) -> None:
    """Print a DataFrame safely."""

    if dataframe.empty:
        print(
            "No observations available for this analysis."
        )
        return

    print(
        dataframe.head(
            max_rows
        ).to_string(
            index=False
        )
    )


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------

def main() -> None:

    print()
    print("=" * 100)
    print(
        "FINSIGHT AI — RELATIONSHIP & NETWORK-LEVEL FRAUD ANALYSIS"
    )
    print("=" * 100)
    print()

    print(
        f"Canonical dataset:\n{CANONICAL_DATASET}"
    )
    print()

    # --------------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------------

    if not CANONICAL_DATASET.exists():

        raise FileNotFoundError(
            "Canonical dataset was not found:\n"
            f"{CANONICAL_DATASET}"
        )

    # --------------------------------------------------------------
    # Load only required columns
    # --------------------------------------------------------------

    print(
        "Loading canonical dataset..."
    )

    required_columns = [
        "nameOrig",
        "nameDest",
        "amount",
        "isFraud",
        "isFlaggedFraud",
    ]

    dataframe = pd.read_parquet(
        CANONICAL_DATASET,
        columns=required_columns,
    )

    print(
        f"Rows    : {len(dataframe):,}"
    )

    print(
        f"Columns : {len(dataframe.columns)}"
    )

    print()

    # --------------------------------------------------------------
    # Population
    # --------------------------------------------------------------

    total_transactions = len(
        dataframe
    )

    total_fraud_transactions = int(
        dataframe[
            "isFraud"
        ].sum()
    )

    overall_fraud_rate = (
        total_fraud_transactions
        / total_transactions
        if total_transactions > 0
        else 0
    )

    origin_entities = int(
        dataframe[
            "nameOrig"
        ].nunique()
    )

    destination_entities = int(
        dataframe[
            "nameDest"
        ].nunique()
    )

    print_header(
        "RELATIONSHIP ANALYSIS POPULATION"
    )

    print(
        f"Total transactions     : "
        f"{total_transactions:,}"
    )

    print(
        f"Fraud transactions     : "
        f"{total_fraud_transactions:,}"
    )

    print(
        f"Fraud rate             : "
        f"{overall_fraud_rate:.6%}"
    )

    print(
        f"Origin entities        : "
        f"{origin_entities:,}"
    )

    print(
        f"Destination entities   : "
        f"{destination_entities:,}"
    )

    # --------------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------------

    print_header(
        "RUNNING RELATIONSHIP UNIQUENESS DIAGNOSTIC"
    )

    analysis = RelationshipAnalysis(
        dataframe
    )

    results = analysis.run()

    relationship_summary = (
        results[
            "relationship_summary"
        ]
    )

    print_dataframe(
        relationship_summary,
        max_rows=20,
    )

    # --------------------------------------------------------------
    # Relationship profile
    # --------------------------------------------------------------

    relationship_profile = (
        results[
            "relationship_profile"
        ]
    )

    print_header(
        "BUILDING RELATIONSHIP PROFILE"
    )

    print(
        f"Unique relationships   : "
        f"{len(relationship_profile):,}"
    )

    print_header(
        "RELATIONSHIP PROFILE"
    )

    print_dataframe(
        relationship_profile,
        max_rows=15,
    )

    # --------------------------------------------------------------
    # Relationship reuse
    # --------------------------------------------------------------

    print_header(
        "RELATIONSHIP REUSE"
    )

    print_dataframe(
        results[
            "relationship_reuse"
        ],
        max_rows=20,
    )

    # --------------------------------------------------------------
    # Origin connectivity
    # --------------------------------------------------------------

    print_header(
        "ORIGIN CONNECTIVITY"
    )

    origin_connectivity = (
        results[
            "origin_connectivity"
        ]
    )

    print(
        origin_connectivity.describe(
            include="all"
        ).to_string()
    )

    # --------------------------------------------------------------
    # Destination connectivity
    # --------------------------------------------------------------

    print_header(
        "DESTINATION CONNECTIVITY"
    )

    destination_connectivity = (
        results[
            "destination_connectivity"
        ]
    )

    print(
        destination_connectivity.describe(
            include="all"
        ).to_string()
    )

    # --------------------------------------------------------------
    # Fraud relationships
    # --------------------------------------------------------------

    print_header(
        "FRAUD-ASSOCIATED RELATIONSHIPS"
    )

    print_dataframe(
        results[
            "fraud_relationship_summary"
        ],
        max_rows=20,
    )

    # --------------------------------------------------------------
    # Concentration
    # --------------------------------------------------------------

    print_header(
        "RELATIONSHIP FRAUD CONCENTRATION"
    )

    print_dataframe(
        results[
            "relationship_fraud_concentration"
        ],
        max_rows=20,
    )

    # --------------------------------------------------------------
    # Connectivity comparison
    # --------------------------------------------------------------

    print_header(
        "CONNECTIVITY COMPARISON"
    )

    print_dataframe(
        results[
            "connectivity_comparison"
        ],
        max_rows=20,
    )

    # --------------------------------------------------------------
    # Save reports
    # --------------------------------------------------------------

    print_header(
        "SAVING RELATIONSHIP ANALYSIS"
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_to_save = {
        "relationship_profile.csv": (
            results[
                "relationship_profile"
            ]
        ),
        "relationship_reuse.csv": (
            results[
                "relationship_reuse"
            ]
        ),
        "origin_connectivity.csv": (
            results[
                "origin_connectivity"
            ]
        ),
        "destination_connectivity.csv": (
            results[
                "destination_connectivity"
            ]
        ),
        "fraud_relationship_summary.csv": (
            results[
                "fraud_relationship_summary"
            ]
        ),
        "relationship_fraud_concentration.csv": (
            results[
                "relationship_fraud_concentration"
            ]
        ),
        "connectivity_comparison.csv": (
            results[
                "connectivity_comparison"
            ]
        ),
        "relationship_summary.csv": (
            results[
                "relationship_summary"
            ]
        ),
        "population_summary.csv": (
            results[
                "population_summary"
            ]
        ),
    }

    for filename, report in (
        files_to_save.items()
    ):

        output_path = (
            OUTPUT_DIRECTORY
            / filename
        )

        report.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------------
    # Completion
    # --------------------------------------------------------------

    print_header(
        "RELATIONSHIP ANALYSIS COMPLETE"
    )

    print(
        "Canonical dataset was not modified."
    )

    print(
        "No ML features were created."
    )

    print(
        "No canonical statistical transformations were applied."
    )


if __name__ == "__main__":
    main()