from __future__ import annotations

import json
from pathlib import Path

from src.data.canonical_lineage import (
    calculate_schema_fingerprint,
    create_canonical_run_id,
    sha256_file,
)


class CanonicalLineageValidationError(Exception):
    """Raised when canonical lineage validation fails."""


def load_manifest(
    manifest_path: Path,
) -> dict:

    if not manifest_path.exists():
        raise CanonicalLineageValidationError(
            f"Manifest does not exist: {manifest_path}"
        )

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise CanonicalLineageValidationError(
            f"Invalid JSON manifest: {exc}"
        ) from exc


def validate_source_hash(
    source_path: Path,
    manifest: dict,
) -> None:

    actual = sha256_file(
        source_path
    )

    expected = manifest[
        "source"
    ]["sha256"]

    if actual != expected:
        raise CanonicalLineageValidationError(
            "Source SHA-256 does not match manifest."
        )


def validate_canonical_hash(
    canonical_path: Path,
    manifest: dict,
) -> None:

    actual = sha256_file(
        canonical_path
    )

    expected = manifest[
        "canonical"
    ]["sha256"]

    if actual != expected:
        raise CanonicalLineageValidationError(
            "Canonical SHA-256 does not match manifest."
        )


def validate_file_size(
    path: Path,
    expected_size: int,
    label: str,
) -> None:

    actual_size = path.stat().st_size

    if actual_size != expected_size:
        raise CanonicalLineageValidationError(
            f"{label} file size mismatch: "
            f"expected={expected_size}, "
            f"actual={actual_size}"
        )


def validate_canonical_schema(
    canonical_path: Path,
    manifest: dict,
) -> None:

    import pandas as pd

    df = pd.read_parquet(
        canonical_path
    )

    actual_schema = (
        calculate_schema_fingerprint(df)
    )

    expected_schema = manifest[
        "canonical"
    ]["schema_sha256"]

    if actual_schema != expected_schema:
        raise CanonicalLineageValidationError(
            "Canonical schema fingerprint "
            "does not match manifest."
        )

    expected_columns = manifest[
        "canonical"
    ]["column_names"]

    if list(df.columns) != expected_columns:
        raise CanonicalLineageValidationError(
            "Canonical column order does not "
            "match manifest."
        )

    expected_rows = manifest[
        "canonical"
    ]["rows"]

    if len(df) != expected_rows:
        raise CanonicalLineageValidationError(
            "Canonical row count does not "
            "match manifest."
        )


def validate_run_identity(
    manifest: dict,
) -> None:

    expected = create_canonical_run_id(
        manifest["source"]["sha256"],
        manifest["canonical"]["sha256"],
        manifest["git"]["commit"],
    )

    actual = manifest[
        "run"
    ]["run_id"]

    if expected != actual:
        raise CanonicalLineageValidationError(
            f"Run ID mismatch: "
            f"expected={expected}, "
            f"actual={actual}"
        )


def validate_canonical_lineage(
    source_path: Path,
    canonical_path: Path,
    manifest_path: Path,
) -> dict:

    manifest = load_manifest(
        manifest_path
    )

    validate_source_hash(
        source_path,
        manifest,
    )

    validate_canonical_hash(
        canonical_path,
        manifest,
    )

    validate_file_size(
        source_path,
        manifest["source"]["file_size_bytes"],
        "Source",
    )

    validate_file_size(
        canonical_path,
        manifest["canonical"]["file_size_bytes"],
        "Canonical",
    )

    validate_canonical_schema(
        canonical_path,
        manifest,
    )

    validate_run_identity(
        manifest
    )

    return manifest