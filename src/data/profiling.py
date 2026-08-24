from typing import Any

import pandas as pd


def get_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(memory_mb, 2),
        "column_names": list(df.columns),
    }


def get_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": df.isna().sum().values,
            "missing_percentage": (
                df.isna().mean().values * 100
            ).round(4),
            "unique_count": df.nunique(dropna=True).values,
        }
    )

    return profile


def get_fraud_profile(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)

    fraud_count = int(df["isFraud"].sum())
    legitimate_count = total - fraud_count

    fraud_rate = (
        fraud_count / total * 100
        if total > 0
        else 0
    )

    fraud_by_type = (
        df.groupby("type")["isFraud"]
        .agg(
            transactions="count",
            fraud_transactions="sum",
            fraud_rate="mean",
        )
        .reset_index()
    )

    fraud_by_type["fraud_rate"] *= 100

    return {
        "total_transactions": total,
        "fraud_transactions": fraud_count,
        "legitimate_transactions": legitimate_count,
        "fraud_rate_percentage": round(fraud_rate, 6),
        "fraud_by_type": fraud_by_type,
    }


def get_transaction_profile(df: pd.DataFrame) -> dict[str, Any]:
    amount = df["amount"]

    amount_statistics = {
        "min": float(amount.min()),
        "max": float(amount.max()),
        "mean": float(amount.mean()),
        "median": float(amount.median()),
        "std": float(amount.std()),
        "q25": float(amount.quantile(0.25)),
        "q75": float(amount.quantile(0.75)),
        "q95": float(amount.quantile(0.95)),
        "q99": float(amount.quantile(0.99)),
    }

    transaction_type_counts = (
        df["type"]
        .value_counts()
        .rename_axis("type")
        .reset_index(name="count")
    )

    return {
        "amount_statistics": amount_statistics,
        "transaction_type_counts": transaction_type_counts,
    }


def get_temporal_profile(df: pd.DataFrame) -> dict[str, Any]:
    step_counts = (
        df.groupby("step")
        .agg(
            transactions=("step", "size"),
            fraud_transactions=("isFraud", "sum"),
        )
        .reset_index()
    )

    step_counts["fraud_rate"] = (
        step_counts["fraud_transactions"]
        / step_counts["transactions"]
        * 100
    )

    return {
        "min_step": int(df["step"].min()),
        "max_step": int(df["step"].max()),
        "unique_steps": int(df["step"].nunique()),
        "transactions_per_step": step_counts,
    }


def get_entity_profile(df: pd.DataFrame) -> dict[str, Any]:
    origin_counts = (
        df["nameOrig"]
        .value_counts()
        .rename_axis("nameOrig")
        .reset_index(name="transaction_count")
    )

    destination_counts = (
        df["nameDest"]
        .value_counts()
        .rename_axis("nameDest")
        .reset_index(name="transaction_count")
    )

    return {
        "unique_origins": int(df["nameOrig"].nunique()),
        "unique_destinations": int(df["nameDest"].nunique()),
        "top_origins": origin_counts.head(20),
        "top_destinations": destination_counts.head(20),
    }


def get_balance_profile(df: pd.DataFrame) -> dict[str, Any]:
    origin_balance_change = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )

    destination_balance_change = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    return {
        "origin_reconciliation_error": {
            "mean": float(origin_balance_change.mean()),
            "median": float(origin_balance_change.median()),
            "max_abs": float(origin_balance_change.abs().max()),
        },
        "destination_reconciliation_error": {
            "mean": float(destination_balance_change.mean()),
            "median": float(destination_balance_change.median()),
            "max_abs": float(destination_balance_change.abs().max()),
        },
    }