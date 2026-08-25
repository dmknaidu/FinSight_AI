from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleaningValidationResult:
    name: str
    status: str
    message: str


class CleaningValidationError(Exception):
    """Raised when cleaning validation cannot be completed."""


def validate_row_counts(
    original: pd.DataFrame,
    canonical: pd.DataFrame,
    quarantine: pd.DataFrame,
) -> CleaningValidationResult:

    original_count = len(original)
    canonical_count = len(canonical)
    quarantine_count = len(quarantine)

    if canonical_count + quarantine_count != original_count:

        return CleaningValidationResult(
            name="Row count preservation",
            status="FAIL",
            message=(
                f"Original={original_count:,}, "
                f"canonical={canonical_count:,}, "
                f"quarantine={quarantine_count:,}"
            ),
        )

    return CleaningValidationResult(
        name="Row count preservation",
        status="PASS",
        message=(
            f"{original_count:,} source rows accounted for."
        ),
    )


def validate_partition_exclusivity(
    canonical: pd.DataFrame,
    quarantine: pd.DataFrame,
) -> CleaningValidationResult:

    overlap = canonical.index.intersection(
        quarantine.index
    )

    if len(overlap) > 0:

        return CleaningValidationResult(
            name="Partition exclusivity",
            status="FAIL",
            message=(
                f"{len(overlap):,} rows appear in "
                "both canonical and quarantine."
            ),
        )

    return CleaningValidationResult(
        name="Partition exclusivity",
        status="PASS",
        message=(
            "Canonical and quarantine partitions "
            "are mutually exclusive."
        ),
    )


def validate_row_coverage(
    original: pd.DataFrame,
    canonical: pd.DataFrame,
    quarantine: pd.DataFrame,
) -> CleaningValidationResult:

    combined_index = (
        canonical.index
        .union(quarantine.index)
    )

    if not combined_index.equals(
        original.index
    ):

        return CleaningValidationResult(
            name="Source row coverage",
            status="FAIL",
            message=(
                "Canonical and quarantine partitions "
                "do not cover exactly the source rows."
            ),
        )

    return CleaningValidationResult(
        name="Source row coverage",
        status="PASS",
        message=(
            "Every source row exists in exactly one "
            "output partition."
        ),
    )


def validate_columns(
    original: pd.DataFrame,
    canonical: pd.DataFrame,
    quarantine: pd.DataFrame,
) -> list[CleaningValidationResult]:

    results = []

    expected_columns = list(
        original.columns
    )

    canonical_columns = list(
        canonical.columns
    )

    if canonical_columns != expected_columns:

        results.append(
            CleaningValidationResult(
                name="Canonical columns",
                status="FAIL",
                message=(
                    "Canonical dataset columns differ "
                    "from the source dataset."
                ),
            )
        )

    else:

        results.append(
            CleaningValidationResult(
                name="Canonical columns",
                status="PASS",
                message=(
                    "Canonical dataset preserves "
                    "the source column structure."
                ),
            )
        )

    quarantine_expected = [
        "quarantine_reason",
        *expected_columns,
    ]

    quarantine_columns = list(
        quarantine.columns
    )

    if len(quarantine) > 0:

        if quarantine_columns != quarantine_expected:

            results.append(
                CleaningValidationResult(
                    name="Quarantine columns",
                    status="FAIL",
                    message=(
                        "Quarantine dataset does not "
                        "contain the expected audit structure."
                    ),
                )
            )

        else:

            results.append(
                CleaningValidationResult(
                    name="Quarantine columns",
                    status="PASS",
                    message=(
                        "Quarantine dataset contains "
                        "audit reason and source columns."
                    ),
                )
            )

    else:

        results.append(
            CleaningValidationResult(
                name="Quarantine columns",
                status="PASS",
                message=(
                    "No quarantine records exist; "
                    "quarantine structure is not populated."
                ),
            )
        )

    return results


def validate_target_preservation(
    original: pd.DataFrame,
    canonical: pd.DataFrame,
    quarantine: pd.DataFrame,
) -> list[CleaningValidationResult]:

    results = []

    for column in [
        "isFraud",
        "isFlaggedFraud",
    ]:

        original_count = int(
            (original[column] == 1).sum()
        )

        canonical_count = int(
            (canonical[column] == 1).sum()
        )

        quarantine_count = int(
            (quarantine[column] == 1).sum()
        ) if column in quarantine.columns else 0

        output_count = (
            canonical_count
            + quarantine_count
        )

        if output_count != original_count:

            results.append(
                CleaningValidationResult(
                    name=f"{column} preservation",
                    status="FAIL",
                    message=(
                        f"Original={original_count:,}, "
                        f"output={output_count:,}"
                    ),
                )
            )

        else:

            results.append(
                CleaningValidationResult(
                    name=f"{column} preservation",
                    status="PASS",
                    message=(
                        f"{original_count:,} positive "
                        f"records preserved."
                    ),
                )
            )

    return results


def validate_canonical_integrity(
    canonical: pd.DataFrame,
) -> list[CleaningValidationResult]:

    results = []

    if canonical.empty:

        results.append(
            CleaningValidationResult(
                name="Canonical dataset",
                status="FAIL",
                message=(
                    "Canonical dataset contains zero rows."
                ),
            )
        )

        return results

    results.append(
        CleaningValidationResult(
            name="Canonical dataset",
            status="PASS",
            message=(
                f"Canonical dataset contains "
                f"{len(canonical):,} rows."
            ),
        )
    )

    missing_values = int(
        canonical.isna().sum().sum()
    )

    if missing_values > 0:

        results.append(
            CleaningValidationResult(
                name="Canonical missing values",
                status="FAIL",
                message=(
                    f"Found {missing_values:,} "
                    "missing values."
                ),
            )
        )

    else:

        results.append(
            CleaningValidationResult(
                name="Canonical missing values",
                status="PASS",
                message=(
                    "No missing values in canonical data."
                ),
            )
        )

    return results


def validate_cleaning(
    original: pd.DataFrame,
    canonical: pd.DataFrame,
    quarantine: pd.DataFrame,
) -> list[CleaningValidationResult]:

    results = []

    results.append(
        validate_row_counts(
            original,
            canonical,
            quarantine,
        )
    )

    results.append(
        validate_partition_exclusivity(
            canonical,
            quarantine,
        )
    )

    results.append(
        validate_row_coverage(
            original,
            canonical,
            quarantine,
        )
    )

    results.extend(
        validate_columns(
            original,
            canonical,
            quarantine,
        )
    )

    results.extend(
        validate_target_preservation(
            original,
            canonical,
            quarantine,
        )
    )

    results.extend(
        validate_canonical_integrity(
            canonical
        )
    )

    return results