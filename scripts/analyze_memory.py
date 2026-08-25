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

from src.data.memory_analysis import (
    build_memory_summary,
    calculate_memory_usage,
    get_numeric_ranges,
    get_string_cardinality,
    estimate_integer_downcasts,
    estimate_float_downcasts,
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
    / "data_engineering"
)


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def save_report(
    df,
    filename: str,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = REPORT_DIR / filename

    df.to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")


def main() -> None:

    print_header(
        "FINSIGHT AI — MEMORY ANALYSIS"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    print("\nLoading dataset...")

    try:
        df, metadata = ingest_csv(
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
    # Overall memory
    # ------------------------------------------------------------------

    summary = build_memory_summary(
        df
    )

    print_header(
        "CURRENT MEMORY SUMMARY"
    )

    print(
        f"Rows          : {summary['rows']:,}"
    )

    print(
        f"Columns       : {summary['columns']}"
    )

    print(
        f"Memory        : "
        f"{summary['memory_mb']:,.2f} MB"
    )

    print(
        f"Memory        : "
        f"{summary['memory_gb']:.3f} GB"
    )

    # ------------------------------------------------------------------
    # Per-column memory
    # ------------------------------------------------------------------

    memory_profile = calculate_memory_usage(
        df
    )

    print_header(
        "PER-COLUMN MEMORY PROFILE"
    )

    print(
        memory_profile.to_string(
            index=False,
            float_format=lambda value:
                f"{value:,.4f}",
        )
    )

    save_report(
        memory_profile,
        "memory_profile.csv",
    )

    # ------------------------------------------------------------------
    # Numeric ranges
    # ------------------------------------------------------------------

    numeric_ranges = get_numeric_ranges(
        df
    )

    print_header(
        "NUMERIC RANGE ANALYSIS"
    )

    print(
        numeric_ranges.to_string(
            index=False,
            float_format=lambda value:
                f"{value:,.6f}",
        )
    )

    save_report(
        numeric_ranges,
        "numeric_ranges.csv",
    )

    # ------------------------------------------------------------------
    # String cardinality
    # ------------------------------------------------------------------

    string_cardinality = get_string_cardinality(
        df
    )

    print_header(
        "STRING CARDINALITY ANALYSIS"
    )

    print(
        string_cardinality.to_string(
            index=False,
            float_format=lambda value:
                f"{value:,.8f}",
        )
    )

    save_report(
        string_cardinality,
        "string_cardinality.csv",
    )

    # ------------------------------------------------------------------
    # Integer downcast candidates
    # ------------------------------------------------------------------

    integer_candidates = (
        estimate_integer_downcasts(df)
    )

    print_header(
        "INTEGER DOWNCAST ANALYSIS"
    )

    if integer_candidates.empty:

        print(
            "No integer columns found."
        )

    else:

        print(
            integer_candidates.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:,.6f}",
            )
        )

    save_report(
        integer_candidates,
        "integer_downcast_candidates.csv",
    )

    # ------------------------------------------------------------------
    # Float downcast candidates
    # ------------------------------------------------------------------

    float_candidates = (
        estimate_float_downcasts(df)
    )

    print_header(
        "FLOAT32 PRECISION ANALYSIS"
    )

    if float_candidates.empty:

        print(
            "No floating-point columns found."
        )

    else:

        print(
            float_candidates.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:,.12f}",
            )
        )

    save_report(
        float_candidates,
        "float_downcast_candidates.csv",
    )

    # ------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------

    print_header(
        "MEMORY ANALYSIS COMPLETE"
    )

    print(
        "No DataFrame transformations were performed."
    )

    print(
        "No dtype changes were applied."
    )

    print(
        "\nThe results will be used to design "
        "the Step 3 optimization strategy."
    )


if __name__ == "__main__":
    main()