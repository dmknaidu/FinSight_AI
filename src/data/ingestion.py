from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class IngestionResult:
    """
    Metadata describing a successful dataset ingestion.
    """

    source_path: Path
    rows: int
    columns: int
    column_names: tuple[str, ...]


class DataIngestionError(Exception):
    """
    Raised when the raw dataset cannot be ingested safely.
    """


REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}


def validate_source_file(file_path: str | Path) -> Path:
    """
    Validate that the requested source file exists and is a CSV.

    No data is loaded or modified here.
    """

    path = Path(file_path)

    if not path.exists():
        raise DataIngestionError(
            f"Source dataset does not exist: {path.resolve()}"
        )

    if not path.is_file():
        raise DataIngestionError(
            f"Source path is not a file: {path.resolve()}"
        )

    if path.suffix.lower() != ".csv":
        raise DataIngestionError(
            f"Expected a CSV file, received: {path.suffix}"
        )

    if path.stat().st_size == 0:
        raise DataIngestionError(
            f"Source dataset is empty: {path.resolve()}"
        )

    return path


def validate_columns(df: pd.DataFrame) -> None:
    """
    Verify that all required PaySim columns are present.
    """

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise DataIngestionError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def ingest_csv(
    file_path: str | Path,
) -> tuple[pd.DataFrame, IngestionResult]:
    """
    Load the raw PaySim CSV into a DataFrame.

    This function intentionally performs no cleaning or transformation.

    Returns
    -------
    tuple[pd.DataFrame, IngestionResult]
        The raw DataFrame and ingestion metadata.
    """

    path = validate_source_file(file_path)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataIngestionError(
            f"Failed to read CSV file: {path.resolve()}"
        ) from exc

    if df.empty:
        raise DataIngestionError(
            "Dataset was loaded successfully but contains zero rows."
        )

    validate_columns(df)

    result = IngestionResult(
        source_path=path.resolve(),
        rows=len(df),
        columns=len(df.columns),
        column_names=tuple(df.columns),
    )

    return df, result