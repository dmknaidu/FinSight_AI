from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class CanonicalLineageError(Exception):
    """Raised when canonical lineage generation fails."""


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    if not path.exists():
        raise CanonicalLineageError(
            f"File does not exist: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def get_git_value(
    project_root: Path,
    args: list[str],
) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except subprocess.CalledProcessError as exc:
        raise CanonicalLineageError(
            f"Git command failed: git {' '.join(args)}"
        ) from exc


def get_git_metadata(
    project_root: Path,
) -> dict:

    commit = get_git_value(
        project_root,
        ["rev-parse", "HEAD"],
    )

    branch = get_git_value(
        project_root,
        ["branch", "--show-current"],
    )

    status = get_git_value(
        project_root,
        ["status", "--porcelain"],
    )

    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def calculate_schema_fingerprint(
    df: pd.DataFrame,
) -> str:

    schema_definition = "\n".join(
        f"{column}:{df[column].dtype}"
        for column in df.columns
    )

    return sha256_text(
        schema_definition
    )


def create_canonical_run_id(
    source_sha256: str,
    canonical_sha256: str,
    git_commit: str,
) -> str:

    identity = (
        f"{source_sha256}:"
        f"{canonical_sha256}:"
        f"{git_commit}"
    )

    return sha256_text(identity)[:16]


def generate_canonical_lineage(
    project_root: Path,
    source_path: Path,
    canonical_path: Path,
    schema_path: Path,
) -> dict:

    if not source_path.exists():
        raise CanonicalLineageError(
            f"Source dataset does not exist: {source_path}"
        )

    if not canonical_path.exists():
        raise CanonicalLineageError(
            f"Canonical dataset does not exist: "
            f"{canonical_path}"
        )

    if not schema_path.exists():
        raise CanonicalLineageError(
            f"Schema file does not exist: {schema_path}"
        )

    canonical_df = pd.read_parquet(
        canonical_path
    )

    source_sha256 = sha256_file(
        source_path
    )

    canonical_sha256 = sha256_file(
        canonical_path
    )

    schema_sha256 = sha256_file(
        schema_path
    )

    canonical_schema_sha256 = (
        calculate_schema_fingerprint(
            canonical_df
        )
    )

    git = get_git_metadata(
        project_root
    )

    run_id = create_canonical_run_id(
        source_sha256,
        canonical_sha256,
        git["commit"],
    )

    return {
        "lineage_version": "1.0",

        "run": {
            "run_id": run_id,
            "phase": "Phase 1",
            "step": (
                "Step 8 — Canonical Dataset Generation"
            ),
            "generated_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        },

        "source": {
            "filename": source_path.name,
            "file_size_bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },

        "canonical": {
            "filename": canonical_path.name,
            "file_size_bytes": canonical_path.stat().st_size,
            "sha256": canonical_sha256,
            "rows": len(canonical_df),
            "columns": len(canonical_df.columns),
            "schema_sha256": (
                canonical_schema_sha256
            ),
            "column_names": list(
                canonical_df.columns
            ),
            "dtypes": {
                column: str(
                    canonical_df[column].dtype
                )
                for column in canonical_df.columns
            },
        },

        "source_schema": {
            "filename": schema_path.name,
            "sha256": schema_sha256,
        },

        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "pyyaml_version": yaml.__version__,
        },

        "git": git,
    }