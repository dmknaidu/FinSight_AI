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

from src.data.lineage import (
    LineageError,
    create_lineage_manifest,
    save_lineage_manifest,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_engineering"
    / "lineage"
    / "lineage_manifest.json"
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
        "FINSIGHT AI — DATA LINEAGE"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    try:

        df, metadata = ingest_csv(
            DATASET_PATH
        )

    except DataIngestionError as exc:

        print(
            "\nDATA INGESTION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print(
        f"Rows: {metadata.rows:,}"
    )

    print(
        f"Columns: {metadata.columns}"
    )

    # --------------------------------------------------------------
    # Generate lineage
    # --------------------------------------------------------------

    print_header(
        "GENERATING LINEAGE METADATA"
    )

    try:

        manifest = (
            create_lineage_manifest(
                dataset_path=DATASET_PATH,
                df=df,
                project_root=PROJECT_ROOT,
                phase="Phase 1",
                step="Step 7 — Data Lineage & Reproducibility",
            )
        )

        save_lineage_manifest(
            manifest,
            OUTPUT_PATH,
        )

    except LineageError as exc:

        print(
            "\nLINEAGE GENERATION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    # --------------------------------------------------------------
    # Display key metadata
    # --------------------------------------------------------------

    print_header(
        "DATASET IDENTITY"
    )

    dataset = manifest[
        "dataset"
    ]

    print(
        f"Filename       : "
        f"{dataset['filename']}"
    )

    print(
        f"File size      : "
        f"{dataset['file_size_bytes']:,} bytes"
    )

    print(
        f"SHA-256        : "
        f"{dataset['sha256']}"
    )

    print_header(
        "SCHEMA IDENTITY"
    )

    schema = manifest[
        "schema"
    ]

    print(
        f"Schema SHA-256 : "
        f"{schema['sha256']}"
    )

    print(
        f"Columns        : "
        f"{len(schema['columns'])}"
    )

    print_header(
        "RUN IDENTITY"
    )

    run = manifest[
        "run"
    ]

    print(
        f"Run ID         : "
        f"{run['run_id']}"
    )

    print(
        f"Phase          : "
        f"{run['phase']}"
    )

    print(
        f"Step           : "
        f"{run['step']}"
    )

    print_header(
        "ENVIRONMENT"
    )

    environment = manifest[
        "environment"
    ]

    print(
        f"Python         : "
        f"{environment['python_version']}"
    )

    print(
        f"Pandas         : "
        f"{environment['pandas_version']}"
    )

    print(
        f"NumPy          : "
        f"{environment['numpy_version']}"
    )

    print(
        f"PyYAML         : "
        f"{environment['pyyaml_version']}"
    )

    print_header(
        "GIT REPRODUCIBILITY"
    )

    git = manifest[
        "git"
    ]

    print(
        f"Commit         : "
        f"{git['commit']}"
    )

    print(
        f"Branch         : "
        f"{git['branch']}"
    )

    print(
        f"Working tree   : "
        f"{'DIRTY' if git['dirty'] else 'CLEAN'}"
    )

    print_header(
        "LINEAGE COMPLETE"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\nNo data was modified."
    )


if __name__ == "__main__":
    main()