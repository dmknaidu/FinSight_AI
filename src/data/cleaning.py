from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleaningResult:
    """Result of applying the cleaning policy."""

    canonical: pd.DataFrame
    quarantine: pd.DataFrame
    total_rows: int
    canonical_rows: int
    quarantine_rows: int


class CleaningError(Exception):
    """Raised when the cleaning process cannot be completed."""


EXPECTED_TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}


FINANCIAL_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def _empty_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """Return a False mask aligned to the DataFrame."""

    return pd.Series(
        False,
        index=df.index,
        dtype=bool,
    )


def build_missing_value_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Identify rows containing missing values.

    Missing values are quarantined because required
    transaction information is unavailable.
    """

    return df.isna().any(axis=1)


def build_invalid_transaction_type_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """Identify rows with unsupported transaction types."""

    if "type" not in df.columns:
        return _empty_mask(df)

    return ~df["type"].isin(
        EXPECTED_TRANSACTION_TYPES
    )


def build_negative_financial_value_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Identify rows containing negative amount or balance
    values.
    """

    mask = _empty_mask(df)

    for column in FINANCIAL_COLUMNS:

        if column not in df.columns:
            continue

        mask |= df[column] < 0

    return mask


def build_invalid_step_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """Identify rows with invalid temporal step values."""

    if "step" not in df.columns:
        return _empty_mask(df)

    return df["step"] < 0


def build_invalid_fraud_flag_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """Identify rows with invalid fraud indicators."""

    mask = _empty_mask(df)

    if "isFraud" in df.columns:

        mask |= ~df["isFraud"].isin(
            {0, 1}
        )

    if "isFlaggedFraud" in df.columns:

        mask |= ~df["isFlaggedFraud"].isin(
            {0, 1}
        )

    return mask


def build_empty_entity_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Identify rows where an origin or destination entity
    identifier is missing or empty.
    """

    mask = _empty_mask(df)

    for column in [
        "nameOrig",
        "nameDest",
    ]:

        if column not in df.columns:
            continue

        values = (
            df[column]
            .astype("string")
        )

        mask |= (
            values.isna()
            | values.str.strip().eq("")
        )

    return mask


def build_quarantine_mask(
    df: pd.DataFrame,
) -> tuple[
    pd.Series,
    dict[str, pd.Series],
]:
    """
    Build the complete row-level quarantine mask.

    Returns:
        combined mask
        individual issue masks
    """

    issue_masks = {
        "missing_required_value":
            build_missing_value_mask(df),

        "invalid_transaction_type":
            build_invalid_transaction_type_mask(df),

        "negative_financial_value":
            build_negative_financial_value_mask(df),

        "invalid_step":
            build_invalid_step_mask(df),

        "invalid_fraud_flag":
            build_invalid_fraud_flag_mask(df),

        "empty_entity_identifier":
            build_empty_entity_mask(df),
    }

    combined = _empty_mask(df)

    for mask in issue_masks.values():
        combined |= mask

    return combined, issue_masks


def build_quarantine_reason(
    index: pd.Index,
    issue_masks: dict[str, pd.Series],
) -> pd.Series:
    """
    Generate an auditable reason for each quarantined row.

    A row can have multiple issues. All applicable issue names
    are retained in the reason field.
    """

    reasons = pd.Series(
        "",
        index=index,
        dtype="string",
    )

    for issue, mask in issue_masks.items():

        affected_index = index[mask]

        for row_index in affected_index:

            current = reasons.loc[row_index]

            if current:
                reasons.loc[row_index] = (
                    f"{current};{issue}"
                )

            else:
                reasons.loc[row_index] = issue

    return reasons


def apply_cleaning_policy(
    df: pd.DataFrame,
) -> CleaningResult:
    """
    Apply the Phase 1 cleaning policy.

    Invalid records are separated into quarantine.
    Valid records remain in the canonical dataset.

    No source DataFrame is modified.
    """

    if df.empty:

        raise CleaningError(
            "Cannot apply cleaning policy to an empty dataset."
        )

    working = df.copy()

    quarantine_mask, issue_masks = (
        build_quarantine_mask(
            working
        )
    )

    canonical = (
        working.loc[
            ~quarantine_mask
        ]
        .copy()
    )

    quarantine = (
        working.loc[
            quarantine_mask
        ]
        .copy()
    )

    if not quarantine.empty:

        quarantine.insert(
            0,
            "quarantine_reason",
            build_quarantine_reason(
                quarantine.index,
                {
                    issue: mask.loc[
                        quarantine.index
                    ]
                    for issue, mask
                    in issue_masks.items()
                },
            ).values,
        )

    canonical_rows = len(
        canonical
    )

    quarantine_rows = len(
        quarantine
    )

    total_rows = len(
        working
    )

    if (
        canonical_rows
        + quarantine_rows
        != total_rows
    ):

        raise CleaningError(
            "Cleaning partition is invalid: "
            "canonical + quarantine does not "
            "equal the original row count."
        )

    return CleaningResult(
        canonical=canonical,
        quarantine=quarantine,
        total_rows=total_rows,
        canonical_rows=canonical_rows,
        quarantine_rows=quarantine_rows,
    )


def validate_cleaning_result(
    original: pd.DataFrame,
    result: CleaningResult,
) -> None:
    """
    Validate the partition produced by cleaning.

    Ensures that every original row belongs to exactly
    one output partition.
    """

    if (
        result.canonical_rows
        + result.quarantine_rows
        != len(original)
    ):
        raise CleaningError(
            "Output row counts do not reconcile "
            "with the original dataset."
        )

    if (
        result.canonical.index.intersection(
            result.quarantine.index
        ).size
        > 0
    ):
        raise CleaningError(
            "A row appears in both canonical "
            "and quarantine datasets."
        )

    combined_index = (
        result.canonical.index
        .union(
            result.quarantine.index
        )
    )

    if not combined_index.equals(
        original.index
    ):
        raise CleaningError(
            "Canonical and quarantine datasets "
            "do not cover the complete original index."
        )