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


from src.data.ingestion import (
    DataIngestionError,
    ingest_csv,
)

from src.data.canonical import (
    CanonicalDatasetError,
    generate_canonical_dataset,
    persist_canonical_dataset,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "finsight_canonical.parquet"
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
        "FINSIGHT AI — CANONICAL DATASET GENERATION"
    )

    print(
        f"Source: {DATASET_PATH}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    print(
        "\nLoading source dataset..."
    )

    try:

        source_df, metadata = ingest_csv(
            DATASET_PATH
        )

    except DataIngestionError as exc:

        print(
            "\nDATA INGESTION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print(
        f"Source rows    : {metadata.rows:,}"
    )

    print(
        f"Source columns : {metadata.columns}"
    )

    # --------------------------------------------------------------
    # Generate canonical dataset
    # --------------------------------------------------------------

    print_header(
        "GENERATING CANONICAL DATASET"
    )

    try:

        canonical_df = (
            generate_canonical_dataset(
                source_df
            )
        )

    except CanonicalDatasetError as exc:

        print(
            "\nCANONICAL GENERATION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    print(
        f"Canonical rows    : "
        f"{len(canonical_df):,}"
    )

    print(
        f"Canonical columns : "
        f"{len(canonical_df.columns)}"
    )

    memory_mb = (
        canonical_df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    print(
        f"Canonical memory  : {memory_mb:.2f} MB"
    )

    print_header(
        "CANONICAL DTYPES"
    )

    print(
        canonical_df.dtypes.to_string()
    )

    # --------------------------------------------------------------
    # Persist
    # --------------------------------------------------------------

    print_header(
        "PERSISTING CANONICAL DATASET"
    )

    try:

        persist_canonical_dataset(
            canonical_df,
            OUTPUT_PATH,
        )

    except CanonicalDatasetError as exc:

        print(
            "\nCANONICAL PERSISTENCE FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print_header(
        "CANONICAL DATASET COMPLETE"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\nSource data was not modified."
    )


if __name__ == "__main__":
    main()