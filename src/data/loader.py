from pathlib import Path

import pandas as pd


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


def load_transactions(file_path: str | Path) -> pd.DataFrame:
    """
    Load the PaySim transaction dataset.

    Parameters
    ----------
    file_path:
        Path to the PaySim CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded transaction dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    ValueError
        If required columns are missing.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {path.resolve()}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, received: {path.suffix}"
        )

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return df