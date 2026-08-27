"""
FinSight AI — Entity-Level Behavioral Analysis

Phase 3, Step 8.

Performs descriptive and statistical analysis of origin and destination
entities without modifying the canonical dataset or creating ML features.

Performance design:
- Reads only columns required by Step 8.
- Uses vectorized fraud_amount construction.
- Uses native pandas groupby aggregations.
- Avoids Python lambdas inside large groupby operations.
- Reuses entity profiles for downstream analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ENTITY_COLUMNS = ("nameOrig", "nameDest")

MIN_TRANSACTION_THRESHOLDS = (1, 2, 3, 5, 10)

CONCENTRATION_PERCENTAGES = (1, 5, 10)

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _safe_median(series: pd.Series) -> float:
    """Return median or NaN for an empty series."""
    if series.empty:
        return float("nan")

    return float(series.median())


def _safe_mann_whitney(
    group_without_fraud: pd.Series,
    group_with_fraud: pd.Series,
) -> tuple[float, float, float]:
    """
    Run a two-sided Mann-Whitney U test.

    Returns:
        U statistic,
        p-value,
        rank-biserial correlation

    The rank-biserial correlation is calculated from the probability that
    a randomly selected fraud-associated entity has a larger value than
    a randomly selected non-fraud-associated entity.
    """

    if group_without_fraud.empty or group_with_fraud.empty:
        return float("nan"), float("nan"), float("nan")

    result = mannwhitneyu(
        group_with_fraud,
        group_without_fraud,
        alternative="two-sided",
        method="asymptotic",
    )

    u_statistic = float(result.statistic)
    p_value = float(result.pvalue)

    n_fraud = len(group_with_fraud)
    n_legitimate = len(group_without_fraud)

    if n_fraud == 0 or n_legitimate == 0:
        return u_statistic, p_value, float("nan")

    probability = u_statistic / (n_fraud * n_legitimate)

    rank_biserial = (2.0 * probability) - 1.0

    return u_statistic, p_value, rank_biserial


# ---------------------------------------------------------------------------
# Entity profiles
# ---------------------------------------------------------------------------

def build_entity_profile(
    df: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """
    Build one behavioral profile row per entity.

    All expensive calculations are delegated to pandas' native
    aggregation machinery.
    """

    required_columns = {
        entity_column,
        "step",
        "amount",
        "isFraud",
        "isFlaggedFraud",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns for {entity_column}: "
            f"{sorted(missing)}"
        )

    # Vectorized fraud monetary exposure.
    #
    # This replaces the previous per-group Python lambda and is
    # substantially faster on millions of rows.
    fraud_amount = df["amount"].where(
        df["isFraud"].eq(1),
        0.0,
    )

    working = pd.DataFrame(
        {
            entity_column: df[entity_column],
            "step": df["step"],
            "amount": df["amount"],
            "isFraud": df["isFraud"],
            "isFlaggedFraud": df["isFlaggedFraud"],
            "fraud_amount": fraud_amount,
        }
    )

    grouped = working.groupby(
        entity_column,
        sort=False,
        observed=True,
    )

    profile = grouped.agg(
        transaction_count=("amount", "size"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        maximum_amount=("amount", "max"),
        fraud_transaction_count=("isFraud", "sum"),
        fraud_amount=("fraud_amount", "sum"),
        flagged_fraud_transaction_count=(
            "isFlaggedFraud",
            "sum",
        ),
        first_step=("step", "min"),
        last_step=("step", "max"),
    ).reset_index()

    # Entity-level fraud rate.
    profile["fraud_rate"] = (
        profile["fraud_transaction_count"]
        / profile["transaction_count"]
    )

    # Temporal activity span.
    profile["temporal_span"] = (
        profile["last_step"] - profile["first_step"]
    )

    # Boolean behavioral indicators represented as uint8.
    profile["has_fraud"] = (
        profile["fraud_transaction_count"] > 0
    ).astype("uint8")

    profile["has_flagged_fraud"] = (
        profile["flagged_fraud_transaction_count"] > 0
    ).astype("uint8")

    return profile


# ---------------------------------------------------------------------------
# Reuse distributions
# ---------------------------------------------------------------------------

def build_reuse_distribution(
    profile: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """
    Quantify entity reuse at predefined transaction-count thresholds.
    """

    transaction_counts = profile["transaction_count"]

    rows = []

    for threshold in MIN_TRANSACTION_THRESHOLDS:
        eligible = transaction_counts >= threshold

        eligible_entities = int(eligible.sum())

        eligible_transactions = int(
            transaction_counts.loc[eligible].sum()
        )

        entity_percentage = (
            eligible_entities / len(profile) * 100.0
            if len(profile)
            else float("nan")
        )

        rows.append(
            {
                "entity_column": entity_column,
                "minimum_transactions": threshold,
                "eligible_entities": eligible_entities,
                "eligible_transaction_count": eligible_transactions,
                "entity_percentage": entity_percentage,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fraud association
# ---------------------------------------------------------------------------

def build_fraud_association(
    profile: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """
    Compare fraud-associated and non-fraud-associated entities.

    Association is evaluated under minimum transaction-count thresholds.
    """

    rows = []

    for threshold in MIN_TRANSACTION_THRESHOLDS:

        eligible = profile[
            profile["transaction_count"] >= threshold
        ]

        with_fraud = eligible[
            eligible["fraud_transaction_count"] > 0
        ]

        without_fraud = eligible[
            eligible["fraud_transaction_count"] == 0
        ]

        eligible_transactions = int(
            eligible["transaction_count"].sum()
        )

        eligible_fraud_transactions = int(
            eligible["fraud_transaction_count"].sum()
        )

        transaction_weighted_fraud_rate = (
            eligible_fraud_transactions / eligible_transactions
            if eligible_transactions
            else float("nan")
        )

        entity_level_fraud_rate = (
            len(with_fraud) / len(eligible)
            if len(eligible)
            else float("nan")
        )

        (
            u_statistic,
            p_value,
            rank_biserial,
        ) = _safe_mann_whitney(
            without_fraud["transaction_count"],
            with_fraud["transaction_count"],
        )

        rows.append(
            {
                "entity_column": entity_column,
                "minimum_transactions": threshold,
                "eligible_entities": int(len(eligible)),
                "entities_with_fraud": int(len(with_fraud)),
                "entities_without_fraud": int(len(without_fraud)),
                "eligible_transactions": eligible_transactions,
                "eligible_fraud_transactions": (
                    eligible_fraud_transactions
                ),
                "entity_level_fraud_rate": (
                    entity_level_fraud_rate
                ),
                "transaction_weighted_fraud_rate": (
                    transaction_weighted_fraud_rate
                ),
                "median_transactions_with_fraud": (
                    _safe_median(
                        with_fraud["transaction_count"]
                    )
                ),
                "median_transactions_without_fraud": (
                    _safe_median(
                        without_fraud["transaction_count"]
                    )
                ),
                "mann_whitney_u": u_statistic,
                "p_value": p_value,
                "rank_biserial_correlation": rank_biserial,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["significant_at_alpha_0_05"] = (
            result["p_value"] < ALPHA
        )

    return result


# ---------------------------------------------------------------------------
# Behavioral comparison
# ---------------------------------------------------------------------------

def build_behavior_comparison(
    profile: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """
    Compare behavioral statistics for fraud-associated and
    non-fraud-associated entities.
    """

    fraud_entities = profile[
        profile["fraud_transaction_count"] > 0
    ]

    legitimate_entities = profile[
        profile["fraud_transaction_count"] == 0
    ]

    behavior_columns = (
        "transaction_count",
        "total_amount",
        "average_amount",
        "median_amount",
        "maximum_amount",
        "temporal_span",
    )

    rows = []

    for column in behavior_columns:

        fraud_values = fraud_entities[column].dropna()

        legitimate_values = legitimate_entities[column].dropna()

        (
            u_statistic,
            p_value,
            rank_biserial,
        ) = _safe_mann_whitney(
            legitimate_values,
            fraud_values,
        )

        legitimate_median = _safe_median(
            legitimate_values
        )

        fraud_median = _safe_median(
            fraud_values
        )

        median_difference = (
            fraud_median - legitimate_median
            if not (
                pd.isna(fraud_median)
                or pd.isna(legitimate_median)
            )
            else float("nan")
        )

        rows.append(
            {
                "entity_column": entity_column,
                "variable": column,
                "legitimate_entity_count": (
                    int(len(legitimate_values))
                ),
                "fraud_associated_entity_count": (
                    int(len(fraud_values))
                ),
                "legitimate_median": legitimate_median,
                "fraud_associated_median": fraud_median,
                "median_difference": median_difference,
                "mann_whitney_u": u_statistic,
                "p_value": p_value,
                "rank_biserial_correlation": rank_biserial,
                "absolute_effect_size": (
                    abs(rank_biserial)
                    if not pd.isna(rank_biserial)
                    else float("nan")
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["significant_at_alpha_0_05"] = (
            result["p_value"] < ALPHA
        )

    return result


# ---------------------------------------------------------------------------
# Fraud concentration
# ---------------------------------------------------------------------------

def build_fraud_concentration(
    profile: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """
    Measure concentration of fraud transactions and fraud monetary
    exposure among the most active entities.

    Entities are ranked by transaction count.

    This function operates entirely on the already-created entity
    profile, avoiding another scan of the transaction-level dataset.
    """

    ranked = profile.sort_values(
        "transaction_count",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    total_entities = len(ranked)

    total_fraud_transactions = int(
        ranked["fraud_transaction_count"].sum()
    )

    total_fraud_amount = float(
        ranked["fraud_amount"].sum()
    )

    rows = []

    for percentage in CONCENTRATION_PERCENTAGES:

        if total_entities == 0:
            top_n = 0
        else:
            top_n = max(
                1,
                int(
                    np.ceil(
                        total_entities
                        * percentage
                        / 100.0
                    )
                ),
            )

        top = ranked.head(top_n)

        top_fraud_transactions = int(
            top["fraud_transaction_count"].sum()
        )

        top_fraud_amount = float(
            top["fraud_amount"].sum()
        )

        rows.append(
            {
                "entity_column": entity_column,
                "ranking_basis": "transaction_count",
                "top_entity_percentage": percentage,
                "top_entity_count": top_n,
                "total_entities": total_entities,
                "total_fraud_transactions": (
                    total_fraud_transactions
                ),
                "top_entities_fraud_transactions": (
                    top_fraud_transactions
                ),
                "fraud_transaction_share": (
                    top_fraud_transactions
                    / total_fraud_transactions
                    if total_fraud_transactions
                    else float("nan")
                ),
                "total_fraud_amount": total_fraud_amount,
                "top_entities_fraud_amount": (
                    top_fraud_amount
                ),
                "fraud_amount_share": (
                    top_fraud_amount
                    / total_fraud_amount
                    if total_fraud_amount
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Complete analysis
# ---------------------------------------------------------------------------

def analyze_entity_behavior(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Execute the complete Step 8 analysis.

    The input dataframe is never modified.
    """

    required_columns = {
        "nameOrig",
        "nameDest",
        "step",
        "amount",
        "isFraud",
        "isFlaggedFraud",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Build profiles once.
    origin_profile = build_entity_profile(
        df,
        "nameOrig",
    )

    destination_profile = build_entity_profile(
        df,
        "nameDest",
    )

    # Reuse distributions.
    origin_reuse = build_reuse_distribution(
        origin_profile,
        "nameOrig",
    )

    destination_reuse = build_reuse_distribution(
        destination_profile,
        "nameDest",
    )

    # Fraud association.
    origin_fraud_association = build_fraud_association(
        origin_profile,
        "nameOrig",
    )

    destination_fraud_association = build_fraud_association(
        destination_profile,
        "nameDest",
    )

    # Fraud concentration is derived from the existing profiles.
    origin_concentration = build_fraud_concentration(
        origin_profile,
        "nameOrig",
    )

    destination_concentration = build_fraud_concentration(
        destination_profile,
        "nameDest",
    )

    # Behavioral comparisons.
    origin_behavior = build_behavior_comparison(
        origin_profile,
        "nameOrig",
    )

    destination_behavior = build_behavior_comparison(
        destination_profile,
        "nameDest",
    )

    behavior_summary = pd.concat(
        [
            origin_behavior,
            destination_behavior,
        ],
        ignore_index=True,
    )

    return {
        "origin_entity_profile": origin_profile,
        "destination_entity_profile": destination_profile,
        "origin_reuse_distribution": origin_reuse,
        "destination_reuse_distribution": destination_reuse,
        "origin_fraud_association": (
            origin_fraud_association
        ),
        "destination_fraud_association": (
            destination_fraud_association
        ),
        "entity_fraud_concentration": pd.concat(
            [
                origin_concentration,
                destination_concentration,
            ],
            ignore_index=True,
        ),
        "entity_behavior_summary": behavior_summary,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_entity_analysis(
    results: dict[str, pd.DataFrame],
    output_directory: str | Path,
) -> None:
    """
    Save all Step 8 analytical reports.
    """

    output_path = Path(output_directory)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, dataframe in results.items():

        dataframe.to_csv(
            output_path / f"{filename}.csv",
            index=False,
        )