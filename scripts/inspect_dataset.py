from pathlib import Path
import sys

import pandas as pd

# Allow importing from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_transactions
from src.data.validation import validate_dataset
from src.data.profiling import (
    get_dataset_overview,
    get_column_profile,
    get_fraud_profile,
    get_transaction_profile,
    get_temporal_profile,
    get_entity_profile,
    get_balance_profile,
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
    / "dataset_forensics"
)


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_validation_results(results) -> None:
    print_header("DATA VALIDATION")

    for result in results:
        print(
            f"[{result.status:<4}] "
            f"{result.check:<35} "
            f"{result.message}"
        )


def save_profile(profile: pd.DataFrame, filename: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / filename
    profile.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


def main() -> None:
    print_header("FINSIGHT AI — DATASET FORENSICS")

    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Place the PaySim CSV inside data/raw/."
        )

    print("\nLoading dataset...")

    df = load_transactions(DATASET_PATH)

    print("Dataset loaded successfully.")

    # ---------------------------------------------------------
    # OVERVIEW
    # ---------------------------------------------------------

    overview = get_dataset_overview(df)

    print_header("DATASET OVERVIEW")

    print(f"Rows          : {overview['rows']:,}")
    print(f"Columns       : {overview['columns']}")
    print(f"Memory Usage  : {overview['memory_mb']:.2f} MB")

    print("\nColumns:")
    for column in overview["column_names"]:
        print(f"  - {column}")

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    validation_results = validate_dataset(df)

    print_validation_results(validation_results)

    # ---------------------------------------------------------
    # COLUMN PROFILE
    # ---------------------------------------------------------

    column_profile = get_column_profile(df)

    print_header("COLUMN PROFILE")

    print(column_profile.to_string(index=False))

    save_profile(
        column_profile,
        "column_profile.csv",
    )

    # ---------------------------------------------------------
    # FRAUD PROFILE
    # ---------------------------------------------------------

    fraud_profile = get_fraud_profile(df)

    print_header("FRAUD PROFILE")

    print(
        f"Total Transactions     : "
        f"{fraud_profile['total_transactions']:,}"
    )

    print(
        f"Fraud Transactions     : "
        f"{fraud_profile['fraud_transactions']:,}"
    )

    print(
        f"Legitimate Transactions: "
        f"{fraud_profile['legitimate_transactions']:,}"
    )

    print(
        f"Fraud Rate             : "
        f"{fraud_profile['fraud_rate_percentage']:.6f}%"
    )

    print("\nFraud by Transaction Type:")

    fraud_by_type = fraud_profile["fraud_by_type"]

    print(
        fraud_by_type.to_string(
            index=False,
            formatters={
                "fraud_rate": "{:.6f}".format
            },
        )
    )

    save_profile(
        fraud_by_type,
        "fraud_by_transaction_type.csv",
    )

    # ---------------------------------------------------------
    # TRANSACTION PROFILE
    # ---------------------------------------------------------

    transaction_profile = get_transaction_profile(df)

    print_header("TRANSACTION PROFILE")

    print("Amount Statistics:")

    for key, value in transaction_profile["amount_statistics"].items():
        print(f"  {key:<10}: {value:,.4f}")

    print("\nTransaction Type Counts:")

    transaction_type_counts = (
        transaction_profile["transaction_type_counts"]
    )

    print(
        transaction_type_counts.to_string(index=False)
    )

    save_profile(
        transaction_type_counts,
        "transaction_type_counts.csv",
    )

    # ---------------------------------------------------------
    # TEMPORAL PROFILE
    # ---------------------------------------------------------

    temporal_profile = get_temporal_profile(df)

    print_header("TEMPORAL PROFILE")

    print(f"Minimum Step : {temporal_profile['min_step']}")
    print(f"Maximum Step : {temporal_profile['max_step']}")
    print(f"Unique Steps : {temporal_profile['unique_steps']}")

    transactions_per_step = (
        temporal_profile["transactions_per_step"]
    )

    print("\nFirst 10 time steps:")

    print(
        transactions_per_step
        .head(10)
        .to_string(index=False)
    )

    save_profile(
        transactions_per_step,
        "transactions_per_step.csv",
    )

    # ---------------------------------------------------------
    # ENTITY PROFILE
    # ---------------------------------------------------------

    entity_profile = get_entity_profile(df)

    print_header("ENTITY PROFILE")

    print(
        f"Unique Origin Entities      : "
        f"{entity_profile['unique_origins']:,}"
    )

    print(
        f"Unique Destination Entities : "
        f"{entity_profile['unique_destinations']:,}"
    )

    print("\nTop Origin Entities:")

    print(
        entity_profile["top_origins"]
        .to_string(index=False)
    )

    print("\nTop Destination Entities:")

    print(
        entity_profile["top_destinations"]
        .to_string(index=False)
    )

    save_profile(
        entity_profile["top_origins"],
        "top_origin_entities.csv",
    )

    save_profile(
        entity_profile["top_destinations"],
        "top_destination_entities.csv",
    )

    # ---------------------------------------------------------
    # BALANCE PROFILE
    # ---------------------------------------------------------

    balance_profile = get_balance_profile(df)

    print_header("BALANCE RECONCILIATION")

    print("Origin balance reconciliation:")

    for key, value in (
        balance_profile["origin_reconciliation_error"]
        .items()
    ):
        print(f"  {key:<10}: {value:,.6f}")

    print("\nDestination balance reconciliation:")

    for key, value in (
        balance_profile["destination_reconciliation_error"]
        .items()
    ):
        print(f"  {key:<10}: {value:,.6f}")

    # ---------------------------------------------------------
    # FINAL MESSAGE
    # ---------------------------------------------------------

    print_header("FORENSICS COMPLETE")

    print(
        "Initial dataset forensics completed successfully."
    )

    print(
        f"Reports saved under:\n"
        f"{REPORT_DIR}"
    )


if __name__ == "__main__":
    main()