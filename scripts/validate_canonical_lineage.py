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


from src.data.canonical_lineage_validation import (
    CanonicalLineageValidationError,
    validate_canonical_lineage,
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

MANIFEST_PATH = (
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
        "FINSIGHT AI — CANONICAL LINEAGE VALIDATION"
    )

    print(
        f"Source    : {SOURCE_PATH}"
    )

    print(
        f"Canonical : {CANONICAL_PATH}"
    )

    print(
        f"Manifest  : {MANIFEST_PATH}"
    )

    print(
        "\nValidating canonical lineage..."
    )

    try:

        manifest = validate_canonical_lineage(
            source_path=SOURCE_PATH,
            canonical_path=CANONICAL_PATH,
            manifest_path=MANIFEST_PATH,
        )

    except CanonicalLineageValidationError as exc:

        print(
            "\nCANONICAL LINEAGE VALIDATION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print(
        "[PASS] Source SHA-256"
    )

    print(
        "[PASS] Canonical SHA-256"
    )

    print(
        "[PASS] Source file size"
    )

    print(
        "[PASS] Canonical file size"
    )

    print(
        "[PASS] Canonical schema fingerprint"
    )

    print(
        "[PASS] Canonical column structure"
    )

    print(
        "[PASS] Canonical row count"
    )

    print(
        "[PASS] Canonical run identity"
    )

    print_header(
        "CANONICAL LINEAGE VALIDATION SUMMARY"
    )

    print(
        "PASS : 8"
    )

    print(
        "FAIL : 0"
    )

    print_header(
        "CANONICAL LINEAGE VALIDATION COMPLETE"
    )

    print(
        f"Source SHA-256    : "
        f"{manifest['source']['sha256']}"
    )

    print(
        f"Canonical SHA-256 : "
        f"{manifest['canonical']['sha256']}"
    )

    print(
        f"Run ID            : "
        f"{manifest['run']['run_id']}"
    )

    print(
        "\nAll canonical lineage checks passed."
    )

    print(
        "No data was modified."
    )


if __name__ == "__main__":
    main()