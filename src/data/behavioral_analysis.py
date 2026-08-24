from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_percentage(numerator: pd.Series | float, denominator: pd.Series | float):
    """
    Safely calculate a percentage without division-by-zero errors.
    """
    return np.where(
        np.asarray(denominator) == 0,
        0.0,
        (np.asarray(numerator) / np.asarray(denominator)) * 100,
    )


# ---------------------------------------------------------------------------
# 1. Fraud vs Transaction Amount
# ---------------------------------------------------------------------------

def analyze_fraud_amount(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compare transaction amount behavior between legitimate and fraudulent
    transactions.

    Returns:
        Dictionary containing:
        - descriptive statistics
        - amount quantiles
        - fraud amount buckets
    """

    amount_statistics = (
        df.groupby("isFraud")["amount"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )

    amount_quantiles = (
        df.groupby("isFraud")["amount"]
        .quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .unstack()
        .reset_index()
    )

    # Use global amount quantiles to create interpretable buckets.
    quantiles = df["amount"].quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )

    bins = [
        -np.inf,
        quantiles.loc[0.25],
        quantiles.loc[0.50],
        quantiles.loc[0.75],
        quantiles.loc[0.90],
        quantiles.loc[0.95],
        quantiles.loc[0.99],
        np.inf,
    ]

    labels = [
        "Q0-Q25",
        "Q25-Q50",
        "Q50-Q75",
        "Q75-Q90",
        "Q90-Q95",
        "Q95-Q99",
        "Q99+",
    ]

    amount_bucket = pd.cut(
        df["amount"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    bucket_analysis = (
        df.assign(amount_bucket=amount_bucket)
        .groupby("amount_bucket", observed=False)
        .agg(
            transactions=("amount", "size"),
            fraud_transactions=("isFraud", "sum"),
        )
        .reset_index()
    )

    bucket_analysis["fraud_rate"] = _safe_percentage(
        bucket_analysis["fraud_transactions"],
        bucket_analysis["transactions"],
    )

    return {
        "statistics": amount_statistics,
        "quantiles": amount_quantiles,
        "bucket_analysis": bucket_analysis,
    }


# ---------------------------------------------------------------------------
# 2. Fraud vs Transaction Type
# ---------------------------------------------------------------------------

def analyze_fraud_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze fraud concentration and fraud rate by transaction type.
    """

    result = (
        df.groupby("type")
        .agg(
            transactions=("type", "size"),
            fraud_transactions=("isFraud", "sum"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
            median_amount=("amount", "median"),
        )
        .reset_index()
    )

    result["legitimate_transactions"] = (
        result["transactions"] - result["fraud_transactions"]
    )

    result["fraud_rate"] = _safe_percentage(
        result["fraud_transactions"],
        result["transactions"],
    )

    result["fraud_share_of_all_fraud"] = _safe_percentage(
        result["fraud_transactions"],
        df["isFraud"].sum(),
    )

    return result.sort_values(
        "fraud_rate",
        ascending=False,
    )


# ---------------------------------------------------------------------------
# 3. Fraud vs Time
# ---------------------------------------------------------------------------

def analyze_fraud_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze transaction volume and fraud behavior over PaySim's step field.
    """

    result = (
        df.groupby("step")
        .agg(
            transactions=("step", "size"),
            fraud_transactions=("isFraud", "sum"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
        )
        .reset_index()
    )

    result["legitimate_transactions"] = (
        result["transactions"] - result["fraud_transactions"]
    )

    result["fraud_rate"] = _safe_percentage(
        result["fraud_transactions"],
        result["transactions"],
    )

    result["fraud_amount"] = (
        df.assign(
            fraud_amount=np.where(
                df["isFraud"] == 1,
                df["amount"],
                0.0,
            )
        )
        .groupby("step")["fraud_amount"]
        .sum()
        .reindex(result["step"])
        .fillna(0)
        .values
    )

    return result


# ---------------------------------------------------------------------------
# 4. Origin Entity Behavior
# ---------------------------------------------------------------------------

def analyze_origin_behavior(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Analyze transaction behavior of origin entities.

    We separate:
    - overall origin activity
    - repeated origins
    - fraud-associated origins
    """

    origin_summary = (
        df.groupby("nameOrig")
        .agg(
            transaction_count=("nameOrig", "size"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
            max_amount=("amount", "max"),
            fraud_transactions=("isFraud", "sum"),
        )
        .reset_index()
    )

    origin_summary["fraud_rate"] = _safe_percentage(
        origin_summary["fraud_transactions"],
        origin_summary["transaction_count"],
    )

    repeat_origin_summary = (
        origin_summary[
            origin_summary["transaction_count"] > 1
        ]
        .sort_values(
            "transaction_count",
            ascending=False,
        )
    )

    fraud_origin_summary = (
        origin_summary[
            origin_summary["fraud_transactions"] > 0
        ]
        .sort_values(
            [
                "fraud_transactions",
                "transaction_count",
            ],
            ascending=False,
        )
    )

    return {
        "all_origins": origin_summary,
        "repeat_origins": repeat_origin_summary,
        "fraud_origins": fraud_origin_summary,
    }


# ---------------------------------------------------------------------------
# 5. Destination Entity Behavior
# ---------------------------------------------------------------------------

def analyze_destination_behavior(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Analyze transaction behavior of destination entities.

    Destination entities tend to be reused more frequently in PaySim,
    so this analysis is important for future graph construction.
    """

    destination_summary = (
        df.groupby("nameDest")
        .agg(
            transaction_count=("nameDest", "size"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
            max_amount=("amount", "max"),
            fraud_transactions=("isFraud", "sum"),
        )
        .reset_index()
    )

    destination_summary["fraud_rate"] = _safe_percentage(
        destination_summary["fraud_transactions"],
        destination_summary["transaction_count"],
    )

    repeat_destination_summary = (
        destination_summary[
            destination_summary["transaction_count"] > 1
        ]
        .sort_values(
            "transaction_count",
            ascending=False,
        )
    )

    fraud_destination_summary = (
        destination_summary[
            destination_summary["fraud_transactions"] > 0
        ]
        .sort_values(
            [
                "fraud_transactions",
                "transaction_count",
            ],
            ascending=False,
        )
    )

    return {
        "all_destinations": destination_summary,
        "repeat_destinations": repeat_destination_summary,
        "fraud_destinations": fraud_destination_summary,
    }


# ---------------------------------------------------------------------------
# 6. Balance Behavior
# ---------------------------------------------------------------------------

def analyze_balance_behavior(df: pd.DataFrame) -> dict[str, Any]:
    """
    Investigate how origin and destination balances change around
    transactions.

    IMPORTANT:
    These are diagnostic calculations only. We are NOT assuming that
    PaySim balances must reconcile perfectly for every transaction type.
    """

    analysis_df = df.copy()

    analysis_df["origin_balance_change"] = (
        analysis_df["oldbalanceOrg"]
        - analysis_df["newbalanceOrig"]
    )

    analysis_df["destination_balance_change"] = (
        analysis_df["newbalanceDest"]
        - analysis_df["oldbalanceDest"]
    )

    analysis_df["origin_expected_new_balance"] = (
        analysis_df["oldbalanceOrg"]
        - analysis_df["amount"]
    )

    analysis_df["destination_expected_new_balance"] = (
        analysis_df["oldbalanceDest"]
        + analysis_df["amount"]
    )

    analysis_df["origin_reconciliation_error"] = (
        analysis_df["newbalanceOrig"]
        - analysis_df["origin_expected_new_balance"]
    )

    analysis_df["destination_reconciliation_error"] = (
        analysis_df["newbalanceDest"]
        - analysis_df["destination_expected_new_balance"]
    )

    summary = (
        analysis_df.groupby(["type", "isFraud"])
        .agg(
            transactions=("amount", "size"),
            average_amount=("amount", "mean"),
            average_origin_balance_change=(
                "origin_balance_change",
                "mean",
            ),
            average_destination_balance_change=(
                "destination_balance_change",
                "mean",
            ),
            median_origin_reconciliation_error=(
                "origin_reconciliation_error",
                "median",
            ),
            median_destination_reconciliation_error=(
                "destination_reconciliation_error",
                "median",
            ),
            mean_abs_origin_reconciliation_error=(
                "origin_reconciliation_error",
                lambda x: x.abs().mean(),
            ),
            mean_abs_destination_reconciliation_error=(
                "destination_reconciliation_error",
                lambda x: x.abs().mean(),
            ),
        )
        .reset_index()
    )

    return {
        "transaction_level": analysis_df,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 7. isFlaggedFraud Analysis
# ---------------------------------------------------------------------------

def analyze_flagged_fraud(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Investigate the relationship between isFlaggedFraud and isFraud.

    This is intentionally diagnostic. We do not yet decide whether
    isFlaggedFraud should be used as an ML feature.
    """

    confusion = pd.crosstab(
        df["isFlaggedFraud"],
        df["isFraud"],
        rownames=["isFlaggedFraud"],
        colnames=["isFraud"],
    )

    flagged_summary = (
        df.groupby("isFlaggedFraud")
        .agg(
            transactions=("isFlaggedFraud", "size"),
            fraud_transactions=("isFraud", "sum"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
        )
        .reset_index()
    )

    flagged_summary["fraud_rate"] = _safe_percentage(
        flagged_summary["fraud_transactions"],
        flagged_summary["transactions"],
    )

    return {
        "confusion": confusion,
        "summary": flagged_summary,
    }


# ---------------------------------------------------------------------------
# 8. Fraud Amount Statistics by Transaction Type
# ---------------------------------------------------------------------------

def analyze_fraud_amount_by_type(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze fraud amount behavior within each transaction type.
    """

    fraud_only = df[df["isFraud"] == 1].copy()

    if fraud_only.empty:
        return pd.DataFrame(
            columns=[
                "type",
                "fraud_transactions",
                "mean_amount",
                "median_amount",
                "min_amount",
                "max_amount",
                "q25_amount",
                "q75_amount",
                "q95_amount",
            ]
        )

    result = (
        fraud_only.groupby("type")["amount"]
        .agg(
            fraud_transactions="count",
            mean_amount="mean",
            median_amount="median",
            min_amount="min",
            max_amount="max",
            q25_amount=lambda x: x.quantile(0.25),
            q75_amount=lambda x: x.quantile(0.75),
            q95_amount=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )

    return result.sort_values(
        "fraud_transactions",
        ascending=False,
    )


# ---------------------------------------------------------------------------
# 9. Behavioral Summary
# ---------------------------------------------------------------------------

def create_behavioral_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Create a compact set of high-level behavioral findings.

    This does not make causal claims. It simply reports observable
    properties of the dataset.
    """

    fraud_count = int(df["isFraud"].sum())

    fraud_types = (
        df[df["isFraud"] == 1]["type"]
        .value_counts()
        .to_dict()
    )

    origin_counts = df["nameOrig"].value_counts()
    destination_counts = df["nameDest"].value_counts()

    return {
        "total_transactions": len(df),
        "fraud_transactions": fraud_count,
        "fraud_rate": (
            fraud_count / len(df) * 100
            if len(df)
            else 0.0
        ),
        "fraud_by_type": fraud_types,
        "origins_with_multiple_transactions": int(
            (origin_counts > 1).sum()
        ),
        "destinations_with_multiple_transactions": int(
            (destination_counts > 1).sum()
        ),
        "max_origin_transactions": int(
            origin_counts.max()
        ),
        "max_destination_transactions": int(
            destination_counts.max()
        ),
    }