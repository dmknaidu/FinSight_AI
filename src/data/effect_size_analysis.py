from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


CONTINUOUS_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TARGET_COLUMN = "isFraud"


class EffectSizeAnalysisError(Exception):
    """Raised when effect-size analysis fails."""


def validate_input(
    df: pd.DataFrame,
) -> None:

    required = CONTINUOUS_COLUMNS + [
        TARGET_COLUMN
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise EffectSizeAnalysisError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not target_values.issubset({0, 1}):
        raise EffectSizeAnalysisError(
            f"Invalid target values: {target_values}"
        )


def calculate_effect_sizes(
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

        if legitimate.empty or fraud.empty:
            raise EffectSizeAnalysisError(
                f"Insufficient observations for "
                f"{column}"
            )

        result = stats.mannwhitneyu(
            fraud,
            legitimate,
            alternative="two-sided",
        )

        u_statistic = float(
            result.statistic
        )

        n_fraud = len(fraud)
        n_legitimate = len(legitimate)

        total_pairs = (
            n_fraud * n_legitimate
        )

        # Probability that a randomly selected
        # fraud observation has a greater rank
        # than a randomly selected legitimate
        # observation.
        probability_fraud_greater = (
            u_statistic / total_pairs
        )

        rank_biserial = (
            2 * probability_fraud_greater
            - 1
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

            median_ratio = (
                fraud_median
                / legitimate_median
            )

        else:

            median_ratio = np.nan

        absolute_effect = abs(
            rank_biserial
        )

        if absolute_effect < 0.10:

            magnitude = (
                "negligible_very_small"
            )

        elif absolute_effect < 0.30:

            magnitude = "small"

        elif absolute_effect < 0.50:

            magnitude = "moderate"

        else:

            magnitude = "large"

        if rank_biserial > 0:

            direction = "fraud_higher"

        elif rank_biserial < 0:

            direction = "fraud_lower"

        else:

            direction = "no_direction"

        records.append(
            {
                "column": column,
                "legitimate_count": n_legitimate,
                "fraud_count": n_fraud,
                "mann_whitney_u": u_statistic,
                "probability_fraud_greater": (
                    probability_fraud_greater
                ),
                "rank_biserial_correlation": (
                    rank_biserial
                ),
                "absolute_effect_size": (
                    absolute_effect
                ),
                "effect_magnitude": magnitude,
                "effect_direction": direction,
                "legitimate_median": (
                    legitimate_median
                ),
                "fraud_median": fraud_median,
                "median_difference": (
                    median_difference
                ),
                "median_ratio": median_ratio,
            }
        )

    return pd.DataFrame(records)


def rank_effect_sizes(
    effect_profile: pd.DataFrame,
) -> pd.DataFrame:

    ranked = effect_profile.copy()

    ranked = ranked.sort_values(
        "absolute_effect_size",
        ascending=False,
    ).reset_index(
        drop=True
    )

    ranked.insert(
        0,
        "effect_rank",
        range(1, len(ranked) + 1),
    )

    return ranked