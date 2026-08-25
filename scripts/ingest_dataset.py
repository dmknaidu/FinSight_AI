from __future__ import annotations

from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.ingestion import (
    DataIngestionError,
    ingest_csv,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_header("FINSIGHT AI — DATA INGESTION")

    print(f"Source dataset:")
    print(f"  {DATASET_PATH}")

    print("\nStarting ingestion...")

    start_time = time.perf_counter()

    try:
        df, metadata = ingest_csv(DATASET_PATH)

    except DataIngestionError as exc:
        print()
        print("INGESTION FAILED")
        print("-" * 80)
        print(str(exc))
        raise SystemExit(1)

    elapsed_time = time.perf_counter() - start_time

    print_header("INGESTION SUCCESSFUL")

    print(f"Source       : {metadata.source_path}")
    print(f"Rows         : {metadata.rows:,}")
    print(f"Columns      : {metadata.columns}")
    print(f"Load time    : {elapsed_time:.2f} seconds")

    print("\nColumns:")

    for index, column in enumerate(
        metadata.column_names,
        start=1,
    ):
        print(f"  {index:2}. {column}")

    print("\nDataFrame shape:")
    print(f"  {df.shape}")

    print("\nFirst 5 rows:")

    print(
        df.head(5).to_string(index=False)
    )

    print_header("INGESTION CHECKPOINT")

    print("✓ Source file exists")
    print("✓ CSV successfully loaded")
    print("✓ Dataset contains rows")
    print("✓ Required columns are present")
    print("✓ Raw values were not transformed")

    print("\nPhase 1 — Step 1 ingestion completed.")


if __name__ == "__main__":
    main()