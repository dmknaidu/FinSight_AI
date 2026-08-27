from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


CONTINUOUS_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TARGET_COLUMN = "isFraud"


class FraudStatisticalComparisonError(Exception):
    """Raised when fraud statistical comparison fails."""


def validate_input(
    df: pd.DataFrame,
) -> None:

    required = (
        CONTINUOUS_COLUMNS
        + [TARGET_COLUMN]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise FraudStatisticalComparisonError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    target_values = set(
        df[TARGET_COLUMN].dropna().unique()
    )

    if not target_values.issubset({0, 1}):
        raise FraudStatisticalComparisonError(
            f"Invalid target values: {target_values}"
        )


def calculate_group_descriptives(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_input(df)

    records = []

    for column in CONTINUOUS_COLUMNS:

        for fraud_status in [0, 1]:

            series = (
                df.loc[
                    df[TARGET_COLUMN] == fraud_status,
                    column,
                ]
                .dropna()
            )

            records.append(
                {
                    "column": column,
                    "group": (
                        "legitimate"
                        if fraud_status == 0
                        else "fraud"
                    ),
                    "isFraud": fraud_status,
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "q25": float(
                        series.quantile(0.25)
                    ),
                    "q75": float(
                        series.quantile(0.75)
                    ),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )

    return pd.DataFrame(records)

'''
def calculate_rank_biserial(
    fraud_values: pd.Series,
    legitimate_values: pd.Series,
    u_statistic: float,
) -> float:

    n_fraud = len(fraud_values)
    n_legitimate = len(
        legitimate_values
    )

    if n_fraud == 0 or n_legitimate == 0:
        return np.nan

    # U is defined here for the fraud group.
    # Rank-biserial correlation:
    #
    # r_rb = 1 - (2U)/(n1*n2)
    #
    # This orientation makes positive values indicate that
    # fraud observations tend to have larger ranks.
    return float(
        1
        - (
            2 * u_statistic
            / (n_fraud * n_legitimate)
        )
    )
'''

def calculate_rank_biserial(
    fraud_values: pd.Series,
    legitimate_values: pd.Series,
    u_statistic: float,
) -> float:

    n_fraud = len(fraud_values)
    n_legitimate = len(legitimate_values)

    if n_fraud == 0 or n_legitimate == 0:
        return np.nan

    # scipy.stats.mannwhitneyu() returns U for the first sample,
    # which here is the fraud group.
    #
    # U1 / (n1 * n2) represents the probability that a randomly
    # selected fraud observation ranks above a randomly selected
    # legitimate observation, with ties contributing 0.5.
    #
    # Rank-biserial correlation:
    #
    # r_rb = (2U1 / (n1*n2)) - 1
    #
    # Positive values therefore indicate that fraud observations
    # tend to have larger values/ranks.
    return float(
        (2 * u_statistic)
        / (n_fraud * n_legitimate)
        - 1
    )

def calculate_mann_whitney_results(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_input(df)

    records = []

    for column in CONTINUOUS_COLUMNS:

        legitimate = (
            df.loc[
                df[TARGET_COLUMN] == 0,
                column,
            ]
            .dropna()
        )

        fraud = (
            df.loc[
                df[TARGET_COLUMN] == 1,
                column,
            ]
            .dropna()
        )

        if fraud.empty or legitimate.empty:
            raise FraudStatisticalComparisonError(
                f"Insufficient group observations "
                f"for {column}"
            )

        result = stats.mannwhitneyu(
            fraud,
            legitimate,
            alternative="two-sided",
        )

        u_statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

        rank_biserial = (
            calculate_rank_biserial(
                fraud,
                legitimate,
                u_statistic,
            )
        )

        fraud_median = float(
            fraud.median()
        )

        legitimate_median = float(
            legitimate.median()
        )

        median_difference = (
            fraud_median
            - legitimate_median
        )

        if legitimate_median != 0:

            median_difference_percentage = (
                median_difference
                / abs(legitimate_median)
                * 100
            )

        else:
            median_difference_percentage = np.nan

        records.append(
            {
                "column": column,
                "legitimate_count": len(
                    legitimate
                ),
                "fraud_count": len(fraud),
                "mann_whitney_u": u_statistic,
                "p_value": p_value,
                "rank_biserial_correlation": (
                    rank_biserial
                ),
                "legitimate_median": (
                    legitimate_median
                ),
                "fraud_median": fraud_median,
                "median_difference": (
                    median_difference
                ),
                "median_difference_percentage": (
                    median_difference_percentage
                ),
            }
        )

    return pd.DataFrame(records)


def apply_fdr_correction(
    results: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:

    corrected = results.copy()

    rejected, adjusted_pvalues, _, _ = (
        multipletests(
            corrected["p_value"].to_numpy(),
            alpha=alpha,
            method="fdr_bh",
        )
    )

    corrected["adjusted_p_value"] = (
        adjusted_pvalues
    )

    corrected[
        "significant_after_fdr"
    ] = rejected

    return corrected


def create_comparison_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    summary = results.copy()
    '''
    summary["effect_direction"] = np.select(
        [
            summary[
                "rank_biserial_correlation"
            ] > 0,
            summary[
                "rank_biserial_correlation"
            ] < 0,
        ],
        [
            "fraud_higher",
            "fraud_lower",
        ],
        default="no_direction",
    )
    '''

    summary["effect_direction"] = np.select(
        [
            summary["rank_biserial_correlation"] > 0,
            summary["rank_biserial_correlation"] < 0,
        ],
        [
            "fraud_higher",
            "fraud_lower",
        ],
        default="no_direction",
    )

    summary["statistical_conclusion"] = np.where(
        summary[
            "significant_after_fdr"
        ],
        "distribution_difference_detected",
        "no_fdr_significant_difference",
    )

    return summary