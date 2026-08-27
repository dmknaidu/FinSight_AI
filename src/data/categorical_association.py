from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


TYPE_COLUMN = "type"
TARGET_COLUMN = "isFraud"


class CategoricalAssociationError(Exception):
    """Raised when categorical association analysis fails."""


def validate_input(df: pd.DataFrame) -> None:

    required = [
        TYPE_COLUMN,
        TARGET_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise CategoricalAssociationError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not target_values.issubset({0, 1}):
        raise CategoricalAssociationError(
            f"Invalid fraud values: {target_values}"
        )


def build_observed_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_input(df)

    table = pd.crosstab(
        df[TYPE_COLUMN],
        df[TARGET_COLUMN],
    )

    table = table.reindex(
        columns=[0, 1],
        fill_value=0,
    )

    table.columns = [
        "legitimate",
        "fraud",
    ]

    return table


def build_expected_table(
    observed: pd.DataFrame,
) -> pd.DataFrame:

    observed_values = observed.to_numpy(
        dtype=float
    )

    grand_total = observed_values.sum()

    if grand_total <= 0:
        raise CategoricalAssociationError(
            "Contingency table is empty."
        )

    row_totals = (
        observed_values.sum(axis=1)
        .reshape(-1, 1)
    )

    column_totals = (
        observed_values.sum(axis=0)
        .reshape(1, -1)
    )

    expected = (
        row_totals
        * column_totals
        / grand_total
    )

    return pd.DataFrame(
        expected,
        index=observed.index,
        columns=observed.columns,
    )


def build_standardized_residuals(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
) -> pd.DataFrame:

    residuals = (
        observed.astype(float)
        - expected
    ) / np.sqrt(expected)

    return residuals


def build_chi_square_contributions(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
) -> pd.DataFrame:

    contributions = (
        (
            observed.astype(float)
            - expected
        ) ** 2
        / expected
    )

    return contributions


def calculate_cramers_v(
    chi2: float,
    observed: pd.DataFrame,
) -> float:

    n = observed.to_numpy().sum()

    rows, columns = observed.shape

    minimum_dimension = min(
        rows - 1,
        columns - 1,
    )

    if n <= 0 or minimum_dimension <= 0:
        return np.nan

    return float(
        np.sqrt(
            chi2
            / (
                n
                * minimum_dimension
            )
        )
    )


def build_category_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_input(df)

    total_fraud = int(
        (df[TARGET_COLUMN] == 1).sum()
    )

    profile = (
        df.groupby(
            TYPE_COLUMN,
            observed=True,
        )[TARGET_COLUMN]
        .agg(
            transactions="size",
            fraud_transactions="sum",
        )
        .reset_index()
    )

    profile["legitimate_transactions"] = (
        profile["transactions"]
        - profile["fraud_transactions"]
    )

    profile["fraud_rate"] = (
        profile["fraud_transactions"]
        / profile["transactions"]
    )

    if total_fraud > 0:
        profile["fraud_composition"] = (
            profile["fraud_transactions"]
            / total_fraud
        )
    else:
        profile["fraud_composition"] = 0.0

    profile["transaction_composition"] = (
        profile["transactions"]
        / profile["transactions"].sum()
    )

    profile = profile.sort_values(
        "fraud_rate",
        ascending=False,
    ).reset_index(drop=True)

    return profile


def run_association_analysis(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:

    validate_input(df)

    observed = build_observed_table(df)

    chi2, p_value, degrees_of_freedom, expected = (
        stats.chi2_contingency(
            observed
        )
    )

    expected_table = pd.DataFrame(
        expected,
        index=observed.index,
        columns=observed.columns,
    )

    residuals = (
        build_standardized_residuals(
            observed,
            expected_table,
        )
    )

    contributions = (
        build_chi_square_contributions(
            observed,
            expected_table,
        )
    )

    cramers_v = calculate_cramers_v(
        float(chi2),
        observed,
    )

    min_expected = float(
        expected.min()
    )

    max_expected = float(
        expected.max()
    )

    contribution_long = (
        contributions.stack()
        .reset_index()
    )

    contribution_long.columns = [
        TYPE_COLUMN,
        "fraud_status",
        "chi_square_contribution",
    ]

    contribution_long[
        "absolute_standardized_residual"
    ] = (
        residuals.stack()
        .abs()
        .to_numpy()
    )

    contribution_long[
        "standardized_residual"
    ] = (
        residuals.stack()
        .to_numpy()
    )

    contribution_long = (
        contribution_long.sort_values(
            "chi_square_contribution",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary = pd.DataFrame(
        [
            {
                "test": (
                    "Pearson chi-square "
                    "test of independence"
                ),
                "chi2_statistic": float(chi2),
                "degrees_of_freedom": int(
                    degrees_of_freedom
                ),
                "p_value": float(p_value),
                "cramers_v": cramers_v,
                "min_expected_frequency": (
                    min_expected
                ),
                "max_expected_frequency": (
                    max_expected
                ),
                "all_expected_frequencies_ge_5": (
                    bool(
                        (expected >= 5).all()
                    )
                ),
                "significant_at_alpha_0_05": (
                    bool(
                        p_value < 0.05
                    )
                ),
                "total_chi_square_contribution": (
                    float(
                        contributions.to_numpy()
                        .sum()
                    )
                ),
            }
        ]
    )

    return {
        "profile": build_category_profile(df),
        "observed": observed,
        "expected": expected_table,
        "residuals": residuals,
        "contributions": contribution_long,
        "summary": summary,
    }