from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.ingestion import (
    DataIngestionError,
    ingest_csv,
)

from src.data.optimization import (
    OptimizationError,
    calculate_optimization_result,
    optimize_dtypes,
    validate_optimization,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — SAFE DTYPE OPTIMIZATION"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    # ------------------------------------------------------------------
    # Load original dataset
    # ------------------------------------------------------------------

    print("\nLoading dataset...")

    try:

        original, metadata = ingest_csv(
            DATASET_PATH
        )

    except DataIngestionError as exc:

        print("\nDATA INGESTION FAILED")
        print("-" * 100)
        print(str(exc))

        raise SystemExit(1)

    print(
        f"Rows: {metadata.rows:,}"
    )

    print(
        f"Columns: {metadata.columns}"
    )

    # ------------------------------------------------------------------
    # Original dtypes
    # ------------------------------------------------------------------

    print_header(
        "ORIGINAL DTYPES"
    )

    print(
        original.dtypes.to_string()
    )

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    print_header(
        "APPLYING APPROVED OPTIMIZATIONS"
    )

    print(
        "step             : int64   → uint16"
    )

    print(
        "type             : object  → category"
    )

    print(
        "isFraud          : int64   → uint8"
    )

    print(
        "isFlaggedFraud   : int64   → uint8"
    )

    print(
        "\nFinancial float columns remain float64."
    )

    print(
        "High-cardinality entity identifiers remain object."
    )

    try:

        optimized = optimize_dtypes(
            original
        )

    except OptimizationError as exc:

        print("\nOPTIMIZATION FAILED")
        print("-" * 100)
        print(str(exc))

        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Optimized dtypes
    # ------------------------------------------------------------------

    print_header(
        "OPTIMIZED DTYPES"
    )

    print(
        optimized.dtypes.to_string()
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    print_header(
        "OPTIMIZATION VALIDATION"
    )

    try:

        validate_optimization(
            original,
            optimized,
        )

    except OptimizationError as exc:

        print("\nOPTIMIZATION VALIDATION FAILED")
        print("-" * 100)
        print(str(exc))

        raise SystemExit(1)

    print(
        "[PASS] Row count preserved"
    )

    print(
        "[PASS] Column count preserved"
    )

    print(
        "[PASS] Column names preserved"
    )

    print(
        "[PASS] Column order preserved"
    )

    print(
        "[PASS] Approved dtypes applied"
    )

    print(
        "[PASS] Numeric ranges preserved"
    )

    print(
        "[PASS] Logical values preserved"
    )

    # ------------------------------------------------------------------
    # Memory results
    # ------------------------------------------------------------------

    result = calculate_optimization_result(
        original,
        optimized,
    )

    print_header(
        "MEMORY OPTIMIZATION RESULTS"
    )

    print(
        f"Original memory    : "
        f"{result.original_memory_mb:,.2f} MB"
    )

    print(
        f"Optimized memory   : "
        f"{result.optimized_memory_mb:,.2f} MB"
    )

    print(
        f"Memory saved       : "
        f"{result.memory_saved_mb:,.2f} MB"
    )

    print(
        f"Memory reduction   : "
        f"{result.memory_reduction_percentage:.2f}%"
    )

    # ------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------

    print_header(
        "SAFE DTYPE OPTIMIZATION COMPLETE"
    )

    print(
        "Optimization completed successfully."
    )

    print(
        "The optimized DataFrame is logically equivalent "
        "to the original dataset."
    )

    print(
        "\nNo output dataset has been persisted yet."
    )

    print(
        "Persistence will be handled in a later "
        "canonical-dataset step."
    )


if __name__ == "__main__":
    main()