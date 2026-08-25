from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


class CanonicalFingerprintError(Exception):
    """Raised when canonical fingerprinting fails."""


def calculate_file_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256 for a file."""

    if not file_path.exists():
        raise CanonicalFingerprintError(
            f"File does not exist: {file_path}"
        )

    sha256 = hashlib.sha256()

    try:
        with file_path.open("rb") as file:

            while True:

                chunk = file.read(chunk_size)

                if not chunk:
                    break

                sha256.update(chunk)

    except OSError as exc:

        raise CanonicalFingerprintError(
            f"Unable to read file: {file_path}"
        ) from exc

    return sha256.hexdigest()


def calculate_schema_fingerprint(
    df: pd.DataFrame,
) -> str:
    """Generate deterministic schema fingerprint."""

    schema_definition = "\n".join(
        f"{column}:{df[column].dtype}"
        for column in df.columns
    )

    return hashlib.sha256(
        schema_definition.encode("utf-8")
    ).hexdigest()


def generate_canonical_fingerprint(
    canonical_path: Path,
) -> dict:
    """
    Generate identity metadata for the persisted
    canonical dataset.
    """

    if not canonical_path.exists():

        raise CanonicalFingerprintError(
            f"Canonical dataset does not exist: "
            f"{canonical_path}"
        )

    try:

        df = pd.read_parquet(
            canonical_path
        )

    except Exception as exc:

        raise CanonicalFingerprintError(
            "Unable to read canonical Parquet dataset."
        ) from exc

    file_sha256 = calculate_file_sha256(
        canonical_path
    )

    schema_sha256 = (
        calculate_schema_fingerprint(df)
    )

    return {
        "filename": canonical_path.name,
        "file_size_bytes": canonical_path.stat().st_size,
        "sha256": file_sha256,
        "schema_sha256": schema_sha256,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {
            column: str(df[column].dtype)
            for column in df.columns
        },
    }