from __future__ import annotations

import numpy as np
import pandas as pd


STEP_COLUMN = "step"
TARGET_COLUMN = "isFraud"
AMOUNT_COLUMN = "amount"

MIN_VOLUME_THRESHOLDS = [100, 1_000, 10_000]


class TemporalAnalysisError(Exception):
    """Raised when temporal analysis fails."""


def validate_input(df: pd.DataFrame) -> None:

    required = [
        STEP_COLUMN,
        TARGET_COLUMN,
        AMOUNT_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise TemporalAnalysisError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if df[STEP_COLUMN].isna().any():
        raise TemporalAnalysisError(
            "Missing step values detected."
        )

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not target_values.issubset({0, 1}):
        raise TemporalAnalysisError(
            f"Invalid fraud values: {target_values}"
        )


def build_temporal_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_input(df)

    grouped = df.groupby(
        STEP_COLUMN,
        observed=True,
    )

    profile = (
        grouped[AMOUNT_COLUMN]
        .agg(
            total_transactions="size",
            total_transaction_amount="sum",
            average_transaction_amount="mean",
        )
        .reset_index()
    )

    fraud = (
        df.loc[df[TARGET_COLUMN] == 1]
        .groupby(
            STEP_COLUMN,
            observed=True,
        )[AMOUNT_COLUMN]
        .agg(
            fraud_transactions="size",
            fraud_amount="sum",
            average_fraud_amount="mean",
        )
        .reset_index()
    )

    profile = profile.merge(
        fraud,
        on=STEP_COLUMN,
        how="left",
    )

    profile["fraud_transactions"] = (
        profile["fraud_transactions"]
        .fillna(0)
        .astype("int64")
    )

    profile["fraud_amount"] = (
        profile["fraud_amount"]
        .fillna(0.0)
    )

    profile["average_fraud_amount"] = (
        profile["average_fraud_amount"]
        .fillna(0.0)
    )

    profile["legitimate_transactions"] = (
        profile["total_transactions"]
        - profile["fraud_transactions"]
    )

    profile["legitimate_amount"] = (
        profile["total_transaction_amount"]
        - profile["fraud_amount"]
    )

    profile["fraud_rate"] = (
        profile["fraud_transactions"]
        / profile["total_transactions"]
    )

    profile["fraud_amount_share"] = np.where(
        profile["total_transaction_amount"] > 0,
        profile["fraud_amount"]
        / profile["total_transaction_amount"],
        0.0,
    )

    profile = profile.sort_values(
        STEP_COLUMN
    ).reset_index(drop=True)

    return profile[
        [
            STEP_COLUMN,
            "total_transactions",
            "legitimate_transactions",
            "fraud_transactions",
            "fraud_rate",
            "total_transaction_amount",
            "legitimate_amount",
            "fraud_amount",
            "fraud_amount_share",
            "average_transaction_amount",
            "average_fraud_amount",
        ]
    ]


def build_temporal_summary(
    profile: pd.DataFrame,
) -> pd.DataFrame:

    total_transactions = int(
        profile["total_transactions"].sum()
    )

    total_fraud = int(
        profile["fraud_transactions"].sum()
    )

    total_fraud_amount = float(
        profile["fraud_amount"].sum()
    )

    total_amount = float(
        profile["total_transaction_amount"].sum()
    )

    overall_fraud_rate = (
        total_fraud / total_transactions
        if total_transactions > 0
        else 0.0
    )

    overall_fraud_amount_share = (
        total_fraud_amount / total_amount
        if total_amount > 0
        else 0.0
    )

    top_volume_step = profile.loc[
        profile["total_transactions"].idxmax()
    ]

    top_fraud_count_step = profile.loc[
        profile["fraud_transactions"].idxmax()
    ]

    top_fraud_rate_step = profile.loc[
        profile["fraud_rate"].idxmax()
    ]

    top_fraud_amount_step = profile.loc[
        profile["fraud_amount"].idxmax()
    ]

    metrics = [
        (
            "observed_steps",
            int(len(profile)),
        ),
        (
            "total_transactions",
            total_transactions,
        ),
        (
            "total_fraud_transactions",
            total_fraud,
        ),
        (
            "overall_fraud_rate",
            overall_fraud_rate,
        ),
        (
            "total_transaction_amount",
            total_amount,
        ),
        (
            "total_fraud_amount",
            total_fraud_amount,
        ),
        (
            "overall_fraud_amount_share",
            overall_fraud_amount_share,
        ),
        (
            "mean_transactions_per_step",
            float(
                profile[
                    "total_transactions"
                ].mean()
            ),
        ),
        (
            "std_transactions_per_step",
            float(
                profile[
                    "total_transactions"
                ].std()
            ),
        ),
        (
            "min_transactions_per_step",
            int(
                profile[
                    "total_transactions"
                ].min()
            ),
        ),
        (
            "max_transactions_per_step",
            int(
                profile[
                    "total_transactions"
                ].max()
            ),
        ),

        # Explicitly labelled as an unweighted
        # descriptive statistic.
        (
            "unweighted_mean_fraud_rate",
            float(
                profile["fraud_rate"].mean()
            ),
        ),
        (
            "median_fraud_rate",
            float(
                profile["fraud_rate"].median()
            ),
        ),
        (
            "std_fraud_rate",
            float(
                profile["fraud_rate"].std()
            ),
        ),
        (
            "min_fraud_rate",
            float(
                profile["fraud_rate"].min()
            ),
        ),
        (
            "max_fraud_rate",
            float(
                profile["fraud_rate"].max()
            ),
        ),
        (
            "steps_with_zero_fraud",
            int(
                (
                    profile[
                        "fraud_transactions"
                    ]
                    == 0
                ).sum()
            ),
        ),
        (
            "steps_with_fraud",
            int(
                (
                    profile[
                        "fraud_transactions"
                    ]
                    > 0
                ).sum()
            ),
        ),
        (
            "steps_with_less_than_10_transactions",
            int(
                (
                    profile[
                        "total_transactions"
                    ]
                    < 10
                ).sum()
            ),
        ),
        (
            "steps_with_less_than_100_transactions",
            int(
                (
                    profile[
                        "total_transactions"
                    ]
                    < 100
                ).sum()
            ),
        ),
        (
            "steps_with_less_than_1000_transactions",
            int(
                (
                    profile[
                        "total_transactions"
                    ]
                    < 1_000
                ).sum()
            ),
        ),
        (
            "steps_with_less_than_10000_transactions",
            int(
                (
                    profile[
                        "total_transactions"
                    ]
                    < 10_000
                ).sum()
            ),
        ),
        (
            "peak_volume_step",
            int(
                top_volume_step[
                    STEP_COLUMN
                ]
            ),
        ),
        (
            "peak_volume",
            int(
                top_volume_step[
                    "total_transactions"
                ]
            ),
        ),
        (
            "peak_fraud_count_step",
            int(
                top_fraud_count_step[
                    STEP_COLUMN
                ]
            ),
        ),
        (
            "peak_fraud_count",
            int(
                top_fraud_count_step[
                    "fraud_transactions"
                ]
            ),
        ),
        (
            "peak_fraud_rate_step",
            int(
                top_fraud_rate_step[
                    STEP_COLUMN
                ]
            ),
        ),
        (
            "peak_fraud_rate",
            float(
                top_fraud_rate_step[
                    "fraud_rate"
                ]
            ),
        ),
        (
            "peak_fraud_amount_step",
            int(
                top_fraud_amount_step[
                    STEP_COLUMN
                ]
            ),
        ),
        (
            "peak_fraud_amount",
            float(
                top_fraud_amount_step[
                    "fraud_amount"
                ]
            ),
        ),
    ]

    return pd.DataFrame(
        metrics,
        columns=[
            "metric",
            "value",
        ],
    )


def build_exposure_threshold_analysis(
    profile: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for threshold in MIN_VOLUME_THRESHOLDS:

        eligible = profile.loc[
            profile["total_transactions"]
            >= threshold
        ]

        total_transactions = int(
            eligible[
                "total_transactions"
            ].sum()
        )

        total_fraud = int(
            eligible[
                "fraud_transactions"
            ].sum()
        )

        fraud_rate = (
            total_fraud
            / total_transactions
            if total_transactions > 0
            else 0.0
        )

        records.append(
            {
                "minimum_transactions_per_step": (
                    threshold
                ),
                "eligible_steps": int(
                    len(eligible)
                ),
                "eligible_transactions": (
                    total_transactions
                ),
                "eligible_fraud_transactions": (
                    total_fraud
                ),
                "exposure_weighted_fraud_rate": (
                    fraud_rate
                ),
                "unweighted_mean_fraud_rate": (
                    float(
                        eligible[
                            "fraud_rate"
                        ].mean()
                    )
                    if len(eligible) > 0
                    else np.nan
                ),
                "median_fraud_rate": (
                    float(
                        eligible[
                            "fraud_rate"
                        ].median()
                    )
                    if len(eligible) > 0
                    else np.nan
                ),
                "min_fraud_rate": (
                    float(
                        eligible[
                            "fraud_rate"
                        ].min()
                    )
                    if len(eligible) > 0
                    else np.nan
                ),
                "max_fraud_rate": (
                    float(
                        eligible[
                            "fraud_rate"
                        ].max()
                    )
                    if len(eligible) > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


def build_temporal_correlations(
    profile: pd.DataFrame,
) -> pd.DataFrame:

    pairs = [
        (
            "transaction_volume",
            "fraud_transactions",
        ),
        (
            "transaction_volume",
            "fraud_rate",
        ),
        (
            "transaction_volume",
            "fraud_amount",
        ),
    ]

    records = []

    from scipy import stats

    for x_column, y_column in pairs:

        x = profile[
            "total_transactions"
        ]

        y = profile[
            {
                "fraud_transactions":
                    "fraud_transactions",
                "fraud_rate":
                    "fraud_rate",
                "fraud_amount":
                    "fraud_amount",
            }[y_column]
        ]

        result = stats.pearsonr(
            x,
            y,
        )

        records.append(
            {
                "x_variable": x_column,
                "y_variable": y_column,
                "pearson_r": float(
                    result.statistic
                ),
                "p_value": float(
                    result.pvalue
                ),
                "observations": len(
                    profile
                ),
            }
        )

    return pd.DataFrame(records)


def build_concentration_analysis(
    profile: pd.DataFrame,
) -> pd.DataFrame:

    top_n = max(
        1,
        int(
            np.ceil(
                len(profile) * 0.10
            )
        ),
    )

    by_fraud_count = (
        profile.sort_values(
            "fraud_transactions",
            ascending=False,
        )
        .head(top_n)
    )

    by_fraud_amount = (
        profile.sort_values(
            "fraud_amount",
            ascending=False,
        )
        .head(top_n)
    )

    total_fraud = int(
        profile["fraud_transactions"].sum()
    )

    total_fraud_amount = float(
        profile["fraud_amount"].sum()
    )

    return pd.DataFrame(
        [
            {
                "measure": "fraud_transactions",
                "top_10_percent_step_count": top_n,
                "total_steps": len(profile),
                "top_10_percent_share": (
                    by_fraud_count[
                        "fraud_transactions"
                    ].sum()
                    / total_fraud
                    if total_fraud > 0
                    else 0.0
                ),
            },
            {
                "measure": "fraud_amount",
                "top_10_percent_step_count": top_n,
                "total_steps": len(profile),
                "top_10_percent_share": (
                    by_fraud_amount[
                        "fraud_amount"
                    ].sum()
                    / total_fraud_amount
                    if total_fraud_amount > 0
                    else 0.0
                ),
            },
        ]
    )


def build_peak_analysis(
    profile: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:

    records = []

    rankings = [
        (
            "transaction_volume",
            "total_transactions",
        ),
        (
            "fraud_count",
            "fraud_transactions",
        ),
        (
            "fraud_rate",
            "fraud_rate",
        ),
        (
            "fraud_amount",
            "fraud_amount",
        ),
    ]

    for ranking_name, column in rankings:

        ranked = (
            profile.sort_values(
                column,
                ascending=False,
            )
            .head(top_n)
        )

        for rank, (_, row) in enumerate(
            ranked.iterrows(),
            start=1,
        ):

            records.append(
                {
                    "ranking": ranking_name,
                    "rank": rank,
                    "step": int(
                        row[STEP_COLUMN]
                    ),
                    "value": float(
                        row[column]
                    ),
                    "total_transactions": int(
                        row[
                            "total_transactions"
                        ]
                    ),
                    "fraud_transactions": int(
                        row[
                            "fraud_transactions"
                        ]
                    ),
                    "fraud_rate": float(
                        row["fraud_rate"]
                    ),
                    "fraud_amount": float(
                        row["fraud_amount"]
                    ),
                }
            )

    return pd.DataFrame(records)