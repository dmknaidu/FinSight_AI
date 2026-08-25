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


from src.data.canonical_fingerprint import (
    CanonicalFingerprintError,
    generate_canonical_fingerprint,
)


CANONICAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "finsight_canonical.parquet"
)


def print_header(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:

    print_header(
        "FINSIGHT AI — CANONICAL DATASET FINGERPRINT"
    )

    print(
        f"Canonical dataset:\n"
        f"{CANONICAL_PATH}"
    )

    print(
        "\nGenerating fingerprint..."
    )

    try:

        fingerprint = (
            generate_canonical_fingerprint(
                CANONICAL_PATH
            )
        )

    except CanonicalFingerprintError as exc:

        print(
            "\nFINGERPRINT GENERATION FAILED"
        )

        print(str(exc))

        raise SystemExit(1)

    print_header(
        "CANONICAL DATASET IDENTITY"
    )

    print(
        f"Filename        : "
        f"{fingerprint['filename']}"
    )

    print(
        f"File size       : "
        f"{fingerprint['file_size_bytes']:,} bytes"
    )

    print(
        f"SHA-256         : "
        f"{fingerprint['sha256']}"
    )

    print(
        f"Schema SHA-256  : "
        f"{fingerprint['schema_sha256']}"
    )

    print(
        f"Rows            : "
        f"{fingerprint['rows']:,}"
    )

    print(
        f"Columns         : "
        f"{fingerprint['columns']}"
    )

    print_header(
        "CANONICAL SCHEMA"
    )

    for column, dtype in (
        fingerprint["dtypes"].items()
    ):

        print(
            f"{column:<20} {dtype}"
        )

    print_header(
        "FINGERPRINT COMPLETE"
    )

    print(
        "Canonical dataset identity generated successfully."
    )

    print(
        "\nNo data was modified."
    )


if __name__ == "__main__":
    main()