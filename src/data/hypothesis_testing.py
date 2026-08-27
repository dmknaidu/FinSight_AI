from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


CONTINUOUS_HYPOTHESES = [
    {
        "hypothesis_id": "H1",
        "variable": "amount",
        "family": "transaction_amount",
    },
    {
        "hypothesis_id": "H2",
        "variable": "oldbalanceOrg",
        "family": "origin_balance",
    },
    {
        "hypothesis_id": "H3",
        "variable": "newbalanceOrig",
        "family": "origin_balance",
    },
    {
        "hypothesis_id": "H4",
        "variable": "oldbalanceDest",
        "family": "destination_balance",
    },
    {
        "hypothesis_id": "H5",
        "variable": "newbalanceDest",
        "family": "destination_balance",
    },
]

TARGET_COLUMN = "isFraud"
TYPE_COLUMN = "type"


class HypothesisTestingError(Exception):
    """Raised when hypothesis testing fails."""


def validate_input(
    df: pd.DataFrame,
) -> None:

    required = [
        item["variable"]
        for item in CONTINUOUS_HYPOTHESES
    ] + [
        TARGET_COLUMN,
        TYPE_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise HypothesisTestingError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not target_values.issubset({0, 1}):
        raise HypothesisTestingError(
            f"Invalid fraud values: {target_values}"
        )


def rank_biserial_correlation(
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

    return float(
        (2 * u_statistic)
        / (n_fraud * n_legitimate)
        - 1
    )


def run_continuous_hypotheses(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_input(df)

    records = []

    for hypothesis in CONTINUOUS_HYPOTHESES:

        variable = hypothesis["variable"]

        legitimate = (
            df.loc[
                df[TARGET_COLUMN] == 0,
                variable,
            ]
            .dropna()
        )

        fraud = (
            df.loc[
                df[TARGET_COLUMN] == 1,
                variable,
            ]
            .dropna()
        )

        if legitimate.empty or fraud.empty:
            raise HypothesisTestingError(
                f"Insufficient observations for "
                f"{variable}"
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

        effect_size = (
            rank_biserial_correlation(
                fraud,
                legitimate,
                u_statistic,
            )
        )

        records.append(
            {
                "hypothesis_id": hypothesis[
                    "hypothesis_id"
                ],
                "family": hypothesis[
                    "family"
                ],
                "variable": variable,
                "test": "Mann-Whitney U",
                "alternative": "two-sided",
                "legitimate_count": len(
                    legitimate
                ),
                "fraud_count": len(fraud),
                "u_statistic": u_statistic,
                "p_value": p_value,
                "rank_biserial_correlation": (
                    effect_size
                ),
            }
        )

    results = pd.DataFrame(records)

    rejected, adjusted_pvalues, _, _ = (
        multipletests(
            results["p_value"].to_numpy(),
            alpha=0.05,
            method="fdr_bh",
        )
    )

    results["adjusted_p_value"] = (
        adjusted_pvalues
    )

    results["significant_after_fdr"] = (
        rejected
    )

    results["effect_direction"] = np.select(
        [
            results[
                "rank_biserial_correlation"
            ] > 0,
            results[
                "rank_biserial_correlation"
            ] < 0,
        ],
        [
            "fraud_higher",
            "fraud_lower",
        ],
        default="no_direction",
    )

    return results


def build_transaction_type_contingency(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_input(df)

    table = pd.crosstab(
        df[TYPE_COLUMN],
        df[TARGET_COLUMN],
    )

    table = table.rename(
        columns={
            0: "legitimate",
            1: "fraud",
        }
    )

    return table


def calculate_cramers_v(
    contingency: pd.DataFrame,
    chi2: float,
) -> float:

    n = contingency.to_numpy().sum()

    if n == 0:
        return np.nan

    rows, columns = (
        contingency.shape
    )

    minimum_dimension = min(
        rows - 1,
        columns - 1,
    )

    if minimum_dimension <= 0:
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


def run_transaction_type_hypothesis(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    validate_input(df)

    contingency = (
        build_transaction_type_contingency(
            df
        )
    )

    chi2, p_value, degrees_of_freedom, expected = (
        stats.chi2_contingency(
            contingency
        )
    )

    expected_df = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns,
    )

    cramers_v = calculate_cramers_v(
        contingency,
        float(chi2),
    )

    result = pd.DataFrame(
        [
            {
                "hypothesis_id": "H6",
                "variable": "type",
                "test": (
                    "Pearson chi-square "
                    "test of independence"
                ),
                "chi2_statistic": float(
                    chi2
                ),
                "degrees_of_freedom": int(
                    degrees_of_freedom
                ),
                "p_value": float(
                    p_value
                ),
                "cramers_v": cramers_v,
                "min_expected_frequency": float(
                    expected.min()
                ),
                "max_expected_frequency": float(
                    expected.max()
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
            }
        ]
    )

    return contingency, result