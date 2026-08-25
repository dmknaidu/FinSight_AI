from __future__ import annotations

import json
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


from src.data.canonical_lineage import (
    CanonicalLineageError,
    generate_canonical_lineage,
)


SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

CANONICAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "finsight_canonical.parquet"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "configs"
    / "schema.yaml"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_engineering"
    / "canonical"
    / "canonical_lineage_manifest.json"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — CANONICAL DATASET LINEAGE"
    )

    print(
        f"Source    : {SOURCE_PATH}"
    )

    print(
        f"Canonical : {CANONICAL_PATH}"
    )

    print(
        f"Schema    : {SCHEMA_PATH}"
    )

    print(
        "\nGenerating canonical lineage..."
    )

    try:

        manifest = generate_canonical_lineage(
            project_root=PROJECT_ROOT,
            source_path=SOURCE_PATH,
            canonical_path=CANONICAL_PATH,
            schema_path=SCHEMA_PATH,
        )

    except CanonicalLineageError as exc:

        print(
            "\nCANONICAL LINEAGE FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    print_header(
        "SOURCE DATASET"
    )

    print(
        f"SHA-256 : "
        f"{manifest['source']['sha256']}"
    )

    print(
        f"Size    : "
        f"{manifest['source']['file_size_bytes']:,} bytes"
    )

    print_header(
        "CANONICAL DATASET"
    )

    print(
        f"SHA-256 : "
        f"{manifest['canonical']['sha256']}"
    )

    print(
        f"Schema  : "
        f"{manifest['canonical']['schema_sha256']}"
    )

    print(
        f"Rows    : "
        f"{manifest['canonical']['rows']:,}"
    )

    print(
        f"Columns : "
        f"{manifest['canonical']['columns']}"
    )

    print(
        f"Size    : "
        f"{manifest['canonical']['file_size_bytes']:,} bytes"
    )

    print_header(
        "RUN IDENTITY"
    )

    print(
        f"Run ID  : "
        f"{manifest['run']['run_id']}"
    )

    print(
        f"Commit  : "
        f"{manifest['git']['commit']}"
    )

    print(
        f"Branch  : "
        f"{manifest['git']['branch']}"
    )

    print(
        f"Dirty   : "
        f"{manifest['git']['dirty']}"
    )

    print_header(
        "CANONICAL LINEAGE COMPLETE"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\nNo data was modified."
    )


if __name__ == "__main__":
    main()