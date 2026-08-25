from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class LineageError(Exception):
    """Raised when lineage metadata cannot be generated."""


@dataclass(frozen=True)
class DatasetFingerprint:
    filename: str
    absolute_path: str
    file_size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SchemaFingerprint:
    columns: list[str]
    dtypes: dict[str, str]
    schema_string: str
    sha256: str


@dataclass(frozen=True)
class DatasetStatistics:
    rows: int
    columns: int
    memory_bytes: int
    memory_mb: float


@dataclass(frozen=True)
class EnvironmentMetadata:
    python_version: str
    platform: str
    pandas_version: str
    numpy_version: str
    pyyaml_version: str


@dataclass(frozen=True)
class GitMetadata:
    commit: str
    branch: str
    dirty: bool


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 hash without loading the complete
    file into memory.
    """

    if not file_path.exists():
        raise LineageError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise LineageError(
            f"Path is not a file: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_dataset(
    file_path: Path,
) -> DatasetFingerprint:
    """
    Generate immutable identity metadata for a dataset file.
    """

    file_path = file_path.resolve()

    return DatasetFingerprint(
        filename=file_path.name,
        absolute_path=str(file_path),
        file_size_bytes=file_path.stat().st_size,
        sha256=calculate_sha256(
            file_path
        ),
    )


def fingerprint_schema(
    df: pd.DataFrame,
) -> SchemaFingerprint:
    """
    Generate a deterministic fingerprint from column names
    and dtypes.
    """

    columns = list(
        df.columns
    )

    dtypes = {
        column: str(
            df[column].dtype
        )
        for column in columns
    }

    schema_string = "|".join(
        f"{column}:{dtypes[column]}"
        for column in columns
    )

    schema_hash = hashlib.sha256(
        schema_string.encode("utf-8")
    ).hexdigest()

    return SchemaFingerprint(
        columns=columns,
        dtypes=dtypes,
        schema_string=schema_string,
        sha256=schema_hash,
    )


def collect_dataset_statistics(
    df: pd.DataFrame,
) -> DatasetStatistics:

    memory_bytes = int(
        df.memory_usage(
            deep=True
        ).sum()
    )

    return DatasetStatistics(
        rows=len(df),
        columns=len(df.columns),
        memory_bytes=memory_bytes,
        memory_mb=round(
            memory_bytes / (1024 ** 2),
            4,
        ),
    )


def collect_environment_metadata() -> EnvironmentMetadata:

    return EnvironmentMetadata(
        python_version=platform.python_version(),
        platform=platform.platform(),
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        pyyaml_version=yaml.__version__,
    )


def _run_git_command(
    args: list[str],
) -> str:

    try:

        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):

        return "unknown"


def collect_git_metadata(
    project_root: Path,
) -> GitMetadata:

    commit = _run_git_command(
        [
            "-C",
            str(project_root),
            "rev-parse",
            "HEAD",
        ]
    )

    branch = _run_git_command(
        [
            "-C",
            str(project_root),
            "branch",
            "--show-current",
        ]
    )

    status = _run_git_command(
        [
            "-C",
            str(project_root),
            "status",
            "--porcelain",
        ]
    )

    return GitMetadata(
        commit=commit,
        branch=branch,
        dirty=bool(status),
    )


def create_run_id(
    dataset_sha256: str,
    git_commit: str,
) -> str:

    payload = (
        f"{dataset_sha256}:{git_commit}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


def create_lineage_manifest(
    dataset_path: Path,
    df: pd.DataFrame,
    project_root: Path,
    phase: str,
    step: str,
) -> dict[str, Any]:

    dataset = fingerprint_dataset(
        dataset_path
    )

    schema = fingerprint_schema(
        df
    )

    statistics = (
        collect_dataset_statistics(
            df
        )
    )

    environment = (
        collect_environment_metadata()
    )

    git = collect_git_metadata(
        project_root
    )

    run_id = create_run_id(
        dataset.sha256,
        git.commit,
    )

    return {
        "lineage_version": "1.0",
        "run": {
            "run_id": run_id,
            "project": "FinSight AI",
            "phase": phase,
            "step": step,
            "started_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        "dataset": asdict(dataset),
        "schema": asdict(schema),
        "statistics": asdict(statistics),
        "environment": asdict(environment),
        "git": asdict(git),
    }


def save_lineage_manifest(
    manifest: dict[str, Any],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import json

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )