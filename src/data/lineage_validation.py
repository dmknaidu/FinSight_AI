from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.lineage import (
    calculate_sha256,
    create_run_id,
    fingerprint_schema,
)


class LineageValidationError(Exception):
    """Raised when lineage validation cannot be completed."""


@dataclass(frozen=True)
class LineageValidationResult:
    name: str
    status: str
    message: str


def load_manifest(
    manifest_path: Path,
) -> dict:
    """Load and validate the lineage manifest structure."""

    if not manifest_path.exists():
        raise LineageValidationError(
            f"Lineage manifest does not exist: "
            f"{manifest_path}"
        )

    try:

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            manifest = json.load(file)

    except json.JSONDecodeError as exc:

        raise LineageValidationError(
            f"Invalid JSON manifest: {exc}"
        ) from exc

    required_sections = {
        "lineage_version",
        "run",
        "dataset",
        "schema",
        "statistics",
        "environment",
        "git",
    }

    missing = (
        required_sections
        - set(manifest.keys())
    )

    if missing:

        raise LineageValidationError(
            "Manifest is missing required sections: "
            + ", ".join(sorted(missing))
        )

    return manifest


def validate_dataset_exists(
    dataset_path: Path,
) -> LineageValidationResult:

    if not dataset_path.exists():

        return LineageValidationResult(
            name="Dataset existence",
            status="FAIL",
            message=(
                f"Dataset does not exist: "
                f"{dataset_path}"
            ),
        )

    if not dataset_path.is_file():

        return LineageValidationResult(
            name="Dataset existence",
            status="FAIL",
            message=(
                f"Dataset path is not a file: "
                f"{dataset_path}"
            ),
        )

    return LineageValidationResult(
        name="Dataset existence",
        status="PASS",
        message="Dataset file exists.",
    )


def validate_file_size(
    dataset_path: Path,
    manifest: dict,
) -> LineageValidationResult:

    expected = int(
        manifest["dataset"][
            "file_size_bytes"
        ]
    )

    actual = dataset_path.stat().st_size

    if actual != expected:

        return LineageValidationResult(
            name="File size",
            status="FAIL",
            message=(
                f"Expected={expected:,} bytes; "
                f"actual={actual:,} bytes."
            ),
        )

    return LineageValidationResult(
        name="File size",
        status="PASS",
        message=(
            f"{actual:,} bytes match manifest."
        ),
    )


def validate_dataset_hash(
    dataset_path: Path,
    manifest: dict,
) -> LineageValidationResult:

    expected = manifest[
        "dataset"
    ]["sha256"]

    actual = calculate_sha256(
        dataset_path
    )

    if actual != expected:

        return LineageValidationResult(
            name="Dataset SHA-256",
            status="FAIL",
            message=(
                "Dataset content hash does not "
                "match the lineage manifest."
            ),
        )

    return LineageValidationResult(
        name="Dataset SHA-256",
        status="PASS",
        message=(
            f"SHA-256 {actual} matches manifest."
        ),
    )


def validate_dataset_statistics(
    df: pd.DataFrame,
    manifest: dict,
) -> list[LineageValidationResult]:

    results = []

    expected_rows = int(
        manifest["statistics"]["rows"]
    )

    expected_columns = int(
        manifest["statistics"]["columns"]
    )

    if len(df) != expected_rows:

        results.append(
            LineageValidationResult(
                name="Row count",
                status="FAIL",
                message=(
                    f"Expected={expected_rows:,}; "
                    f"actual={len(df):,}."
                ),
            )
        )

    else:

        results.append(
            LineageValidationResult(
                name="Row count",
                status="PASS",
                message=(
                    f"{len(df):,} rows match manifest."
                ),
            )
        )

    if len(df.columns) != expected_columns:

        results.append(
            LineageValidationResult(
                name="Column count",
                status="FAIL",
                message=(
                    f"Expected={expected_columns}; "
                    f"actual={len(df.columns)}."
                ),
            )
        )

    else:

        results.append(
            LineageValidationResult(
                name="Column count",
                status="PASS",
                message=(
                    f"{len(df.columns)} columns "
                    "match manifest."
                ),
            )
        )

    return results


def validate_column_order(
    df: pd.DataFrame,
    manifest: dict,
) -> LineageValidationResult:

    expected = manifest[
        "schema"
    ]["columns"]

    actual = list(
        df.columns
    )

    if actual != expected:

        return LineageValidationResult(
            name="Column order",
            status="FAIL",
            message=(
                f"Expected={expected}; "
                f"actual={actual}."
            ),
        )

    return LineageValidationResult(
        name="Column order",
        status="PASS",
        message=(
            "Column names and order match manifest."
        ),
    )


def validate_schema_fingerprint(
    df: pd.DataFrame,
    manifest: dict,
) -> LineageValidationResult:

    fingerprint = fingerprint_schema(
        df
    )

    expected = manifest[
        "schema"
    ]["sha256"]

    if fingerprint.sha256 != expected:

        return LineageValidationResult(
            name="Schema fingerprint",
            status="FAIL",
            message=(
                "Schema SHA-256 does not match "
                "the lineage manifest."
            ),
        )

    return LineageValidationResult(
        name="Schema fingerprint",
        status="PASS",
        message=(
            f"Schema SHA-256 "
            f"{fingerprint.sha256} matches manifest."
        ),
    )


def validate_schema_dtypes(
    df: pd.DataFrame,
    manifest: dict,
) -> LineageValidationResult:

    expected = manifest[
        "schema"
    ]["dtypes"]

    actual = {
        column: str(
            df[column].dtype
        )
        for column in df.columns
    }

    if actual != expected:

        differences = []

        for column in expected:

            if expected.get(column) != actual.get(
                column
            ):

                differences.append(
                    f"{column}: "
                    f"expected={expected.get(column)}, "
                    f"actual={actual.get(column)}"
                )

        return LineageValidationResult(
            name="Schema dtypes",
            status="FAIL",
            message=(
                "; ".join(differences)
            ),
        )

    return LineageValidationResult(
        name="Schema dtypes",
        status="PASS",
        message=(
            "All column dtypes match manifest."
        ),
    )


def validate_run_identity(
    manifest: dict,
) -> LineageValidationResult:

    dataset_hash = manifest[
        "dataset"
    ]["sha256"]

    git_commit = manifest[
        "git"
    ]["commit"]

    expected_run_id = create_run_id(
        dataset_hash,
        git_commit,
    )

    actual_run_id = manifest[
        "run"
    ]["run_id"]

    if expected_run_id != actual_run_id:

        return LineageValidationResult(
            name="Run identity",
            status="FAIL",
            message=(
                f"Expected={expected_run_id}; "
                f"manifest={actual_run_id}."
            ),
        )

    return LineageValidationResult(
        name="Run identity",
        status="PASS",
        message=(
            f"Run ID {actual_run_id} is reproducible."
        ),
    )


def validate_environment_metadata(
    manifest: dict,
) -> LineageValidationResult:

    environment = manifest.get(
        "environment",
        {},
    )

    required = {
        "python_version",
        "platform",
        "pandas_version",
        "numpy_version",
        "pyyaml_version",
    }

    missing = (
        required
        - set(environment.keys())
    )

    if missing:

        return LineageValidationResult(
            name="Environment metadata",
            status="FAIL",
            message=(
                "Missing environment metadata: "
                + ", ".join(sorted(missing))
            ),
        )

    return LineageValidationResult(
        name="Environment metadata",
        status="PASS",
        message=(
            "Python and dependency version metadata "
            "are present."
        ),
    )


def validate_git_metadata(
    manifest: dict,
) -> LineageValidationResult:

    git = manifest.get(
        "git",
        {},
    )

    required = {
        "commit",
        "branch",
        "dirty",
    }

    missing = (
        required
        - set(git.keys())
    )

    if missing:

        return LineageValidationResult(
            name="Git metadata",
            status="FAIL",
            message=(
                "Missing Git metadata: "
                + ", ".join(sorted(missing))
            ),
        )

    return LineageValidationResult(
        name="Git metadata",
        status="PASS",
        message=(
            f"Commit={git['commit']}; "
            f"branch={git['branch']}; "
            f"dirty={git['dirty']}."
        ),
    )


def validate_lineage(
    dataset_path: Path,
    df: pd.DataFrame,
    manifest: dict,
) -> list[LineageValidationResult]:

    results = []

    results.append(
        validate_dataset_exists(
            dataset_path
        )
    )

    results.append(
        validate_file_size(
            dataset_path,
            manifest,
        )
    )

    results.append(
        validate_dataset_hash(
            dataset_path,
            manifest,
        )
    )

    results.extend(
        validate_dataset_statistics(
            df,
            manifest,
        )
    )

    results.append(
        validate_column_order(
            df,
            manifest,
        )
    )

    results.append(
        validate_schema_fingerprint(
            df,
            manifest,
        )
    )

    results.append(
        validate_schema_dtypes(
            df,
            manifest,
        )
    )

    results.append(
        validate_run_identity(
            manifest,
        )
    )

    results.append(
        validate_environment_metadata(
            manifest,
        )
    )

    results.append(
        validate_git_metadata(
            manifest,
        )
    )

    return results