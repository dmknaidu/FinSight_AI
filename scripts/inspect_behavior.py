from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.loader import load_transactions
from src.data.behavioral_analysis import (
    analyze_fraud_amount,
    analyze_fraud_by_type,
    analyze_fraud_over_time,
    analyze_origin_behavior,
    analyze_destination_behavior,
    analyze_balance_behavior,
    analyze_flagged_fraud,
    analyze_fraud_amount_by_type,
    create_behavioral_summary,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "dataset_forensics"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def save_dataframe(df, filename: str) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = REPORT_DIR / filename

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def create_fraud_amount_plot(df) -> None:
    """
    Plot transaction amount distributions using log1p transformation.

    The raw amount distribution is extremely right-skewed, so log1p
    makes the distribution easier to inspect visually.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    legitimate = df.loc[
        df["isFraud"] == 0,
        "amount",
    ]

    fraud = df.loc[
        df["isFraud"] == 1,
        "amount",
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        legitimate.apply(lambda x: __import__("numpy").log1p(x)),
        bins=100,
        alpha=0.6,
        label="Legitimate",
    )

    plt.hist(
        fraud.apply(lambda x: __import__("numpy").log1p(x)),
        bins=100,
        alpha=0.6,
        label="Fraud",
    )

    plt.xlabel("log1p(Transaction Amount)")
    plt.ylabel("Transaction Count")
    plt.title("Transaction Amount Distribution: Fraud vs Legitimate")
    plt.legend()
    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "fraud_amount_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")


def create_fraud_type_plot(type_analysis) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        type_analysis["type"],
        type_analysis["fraud_rate"],
    )

    plt.xlabel("Transaction Type")
    plt.ylabel("Fraud Rate (%)")
    plt.title("Fraud Rate by Transaction Type")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "fraud_by_transaction_type.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")


def create_temporal_plots(time_analysis) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Transaction volume
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        time_analysis["step"],
        time_analysis["transactions"],
    )

    plt.xlabel("Step")
    plt.ylabel("Transaction Count")
    plt.title("Transaction Volume Over Time")
    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "transaction_volume_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")

    # ---------------------------------------------------------
    # Fraud rate
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        time_analysis["step"],
        time_analysis["fraud_rate"],
    )

    plt.xlabel("Step")
    plt.ylabel("Fraud Rate (%)")
    plt.title("Fraud Rate Over Time")
    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "fraud_rate_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print_header(
        "FINSIGHT AI — BEHAVIORAL FORENSICS"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("\nLoading dataset...")

    df = load_transactions(
        DATASET_PATH
    )

    print(
        f"Loaded {len(df):,} transactions."
    )

    # -----------------------------------------------------------------------
    # High-level behavioral summary
    # -----------------------------------------------------------------------

    summary = create_behavioral_summary(df)

    print_header(
        "HIGH-LEVEL BEHAVIORAL SUMMARY"
    )

    print(
        f"Total Transactions                : "
        f"{summary['total_transactions']:,}"
    )

    print(
        f"Fraud Transactions                : "
        f"{summary['fraud_transactions']:,}"
    )

    print(
        f"Fraud Rate                        : "
        f"{summary['fraud_rate']:.6f}%"
    )

    print(
        f"Origins With Multiple Transactions: "
        f"{summary['origins_with_multiple_transactions']:,}"
    )

    print(
        f"Destinations With Multiple Txns   : "
        f"{summary['destinations_with_multiple_transactions']:,}"
    )

    print(
        f"Maximum Origin Transactions       : "
        f"{summary['max_origin_transactions']:,}"
    )

    print(
        f"Maximum Destination Transactions  : "
        f"{summary['max_destination_transactions']:,}"
    )

    # -----------------------------------------------------------------------
    # Fraud vs Amount
    # -----------------------------------------------------------------------

    amount_analysis = analyze_fraud_amount(
        df
    )

    print_header(
        "FRAUD VS TRANSACTION AMOUNT"
    )

    print("Descriptive Statistics:")

    print(
        amount_analysis["statistics"].to_string(
            index=False
        )
    )

    print("\nAmount Quantiles:")

    print(
        amount_analysis["quantiles"].to_string(
            index=False
        )
    )

    print("\nAmount Bucket Analysis:")

    print(
        amount_analysis["bucket_analysis"].to_string(
            index=False
        )
    )

    save_dataframe(
        amount_analysis["statistics"],
        "fraud_amount_statistics.csv",
    )

    save_dataframe(
        amount_analysis["quantiles"],
        "fraud_amount_quantiles.csv",
    )

    save_dataframe(
        amount_analysis["bucket_analysis"],
        "fraud_amount_buckets.csv",
    )

    # -----------------------------------------------------------------------
    # Fraud by Type
    # -----------------------------------------------------------------------

    type_analysis = analyze_fraud_by_type(
        df
    )

    print_header(
        "FRAUD BY TRANSACTION TYPE"
    )

    print(
        type_analysis.to_string(
            index=False
        )
    )

    save_dataframe(
        type_analysis,
        "behavioral_fraud_by_type.csv",
    )

    # -----------------------------------------------------------------------
    # Fraud Amount by Type
    # -----------------------------------------------------------------------

    fraud_amount_type = (
        analyze_fraud_amount_by_type(df)
    )

    print_header(
        "FRAUD AMOUNT BY TRANSACTION TYPE"
    )

    print(
        fraud_amount_type.to_string(
            index=False
        )
    )

    save_dataframe(
        fraud_amount_type,
        "fraud_amount_by_type.csv",
    )

    # -----------------------------------------------------------------------
    # Time Analysis
    # -----------------------------------------------------------------------

    time_analysis = analyze_fraud_over_time(
        df
    )

    print_header(
        "TEMPORAL BEHAVIOR"
    )

    print(
        f"Minimum Step: "
        f"{time_analysis['step'].min()}"
    )

    print(
        f"Maximum Step: "
        f"{time_analysis['step'].max()}"
    )

    print(
        f"Total Steps: "
        f"{time_analysis['step'].nunique()}"
    )

    print("\nHighest Fraud Rate Steps:")

    print(
        time_analysis
        .sort_values(
            "fraud_rate",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nHighest Fraud Volume Steps:")

    print(
        time_analysis
        .sort_values(
            "fraud_transactions",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    save_dataframe(
        time_analysis,
        "fraud_over_time.csv",
    )

    # -----------------------------------------------------------------------
    # Origin Analysis
    # -----------------------------------------------------------------------

    origin_analysis = analyze_origin_behavior(
        df
    )

    print_header(
        "ORIGIN ENTITY BEHAVIOR"
    )

    print(
        f"Total origin entities: "
        f"{len(origin_analysis['all_origins']):,}"
    )

    print(
        f"Repeated origin entities: "
        f"{len(origin_analysis['repeat_origins']):,}"
    )

    print(
        f"Fraud-associated origin entities: "
        f"{len(origin_analysis['fraud_origins']):,}"
    )

    print("\nTop Repeated Origins:")

    print(
        origin_analysis["repeat_origins"]
        .head(20)
        .to_string(index=False)
    )

    print("\nFraud-associated Origins:")

    print(
        origin_analysis["fraud_origins"]
        .head(20)
        .to_string(index=False)
    )

    save_dataframe(
        origin_analysis["all_origins"],
        "origin_entity_behavior.csv",
    )

    save_dataframe(
        origin_analysis["repeat_origins"],
        "repeat_origin_entities.csv",
    )

    save_dataframe(
        origin_analysis["fraud_origins"],
        "fraud_origin_entities.csv",
    )

    # -----------------------------------------------------------------------
    # Destination Analysis
    # -----------------------------------------------------------------------

    destination_analysis = (
        analyze_destination_behavior(df)
    )

    print_header(
        "DESTINATION ENTITY BEHAVIOR"
    )

    print(
        f"Total destination entities: "
        f"{len(destination_analysis['all_destinations']):,}"
    )

    print(
        f"Repeated destination entities: "
        f"{len(destination_analysis['repeat_destinations']):,}"
    )

    print(
        f"Fraud-associated destination entities: "
        f"{len(destination_analysis['fraud_destinations']):,}"
    )

    print("\nTop Repeated Destinations:")

    print(
        destination_analysis[
            "repeat_destinations"
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nFraud-associated Destinations:")

    print(
        destination_analysis[
            "fraud_destinations"
        ]
        .head(20)
        .to_string(index=False)
    )

    save_dataframe(
        destination_analysis["all_destinations"],
        "destination_entity_behavior.csv",
    )

    save_dataframe(
        destination_analysis[
            "repeat_destinations"
        ],
        "repeat_destination_entities.csv",
    )

    save_dataframe(
        destination_analysis[
            "fraud_destinations"
        ],
        "fraud_destination_entities.csv",
    )

    # -----------------------------------------------------------------------
    # Balance Analysis
    # -----------------------------------------------------------------------

    balance_analysis = analyze_balance_behavior(
        df
    )

    print_header(
        "BALANCE BEHAVIOR"
    )

    print(
        balance_analysis["summary"].to_string(
            index=False
        )
    )

    save_dataframe(
        balance_analysis["summary"],
        "balance_behavior_summary.csv",
    )

    # -----------------------------------------------------------------------
    # isFlaggedFraud Analysis
    # -----------------------------------------------------------------------

    flagged_analysis = analyze_flagged_fraud(
        df
    )

    print_header(
        "isFLAGGEDFRAUD ANALYSIS"
    )

    print("Summary:")

    print(
        flagged_analysis["summary"].to_string(
            index=False
        )
    )

    print("\nCross-tabulation:")

    print(
        flagged_analysis["confusion"].to_string()
    )

    save_dataframe(
        flagged_analysis["summary"],
        "flagged_fraud_summary.csv",
    )

    flagged_crosstab_path = (
        REPORT_DIR
        / "flagged_fraud_crosstab.csv"
    )

    flagged_analysis["confusion"].to_csv(
        flagged_crosstab_path
    )

    print(
        f"Saved: {flagged_crosstab_path}"
    )

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------

    print_header(
        "GENERATING VISUALIZATIONS"
    )

    create_fraud_amount_plot(df)

    create_fraud_type_plot(
        type_analysis
    )

    create_temporal_plots(
        time_analysis
    )

    # -----------------------------------------------------------------------
    # Complete
    # -----------------------------------------------------------------------

    print_header(
        "BEHAVIORAL FORENSICS COMPLETE"
    )

    print(
        "All behavioral analysis reports were generated."
    )

    print(
        f"\nReports directory:\n"
        f"{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()