from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class QualityReportError(Exception):
    """Raised when quality report generation fails."""


class DataQualityReport:
    """
    Generate descriptive data-quality reports.

    This component performs analysis only.
    It does not modify the input DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
    ) -> None:

        if df.empty:
            raise QualityReportError(
                "Cannot generate a quality report "
                "for an empty dataset."
            )

        self.df = df

    # ------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------

    def dataset_summary(
        self,
    ) -> pd.DataFrame:

        numeric_columns = (
            self.df.select_dtypes(
                include=np.number
            ).columns
        )

        categorical_columns = (
            self.df.select_dtypes(
                include=[
                    "object",
                    "category",
                    "string",
                ]
            ).columns
        )

        return pd.DataFrame(
            [
                {
                    "metric": "rows",
                    "value": len(self.df),
                },
                {
                    "metric": "columns",
                    "value": len(self.df.columns),
                },
                {
                    "metric": "memory_mb",
                    "value": round(
                        self.df.memory_usage(
                            deep=True
                        ).sum()
                        / (1024 ** 2),
                        4,
                    ),
                },
                {
                    "metric": "numeric_columns",
                    "value": len(
                        numeric_columns
                    ),
                },
                {
                    "metric": "categorical_or_string_columns",
                    "value": len(
                        categorical_columns
                    ),
                },
            ]
        )

    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------

    def completeness_profile(
        self,
    ) -> pd.DataFrame:

        rows = []

        total_rows = len(self.df)

        for column in self.df.columns:

            missing = int(
                self.df[column]
                .isna()
                .sum()
            )

            non_null = (
                total_rows
                - missing
            )

            percentage = (
                missing
                / total_rows
                * 100
            )

            rows.append(
                {
                    "column": column,
                    "dtype": str(
                        self.df[column].dtype
                    ),
                    "row_count": total_rows,
                    "non_null_count": non_null,
                    "missing_count": missing,
                    "missing_percentage": round(
                        percentage,
                        6,
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Cardinality
    # ------------------------------------------------------------------

    def cardinality_profile(
        self,
    ) -> pd.DataFrame:

        rows = []

        total_rows = len(self.df)

        for column in self.df.columns:

            unique_count = int(
                self.df[column]
                .nunique(
                    dropna=True
                )
            )

            ratio = (
                unique_count
                / total_rows
            )

            rows.append(
                {
                    "column": column,
                    "dtype": str(
                        self.df[column].dtype
                    ),
                    "unique_count": unique_count,
                    "cardinality_ratio": round(
                        ratio,
                        8,
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Numeric profile
    # ------------------------------------------------------------------

    def numeric_profile(
        self,
    ) -> pd.DataFrame:

        rows = []

        numeric_columns = (
            self.df.select_dtypes(
                include=np.number
            ).columns
        )

        for column in numeric_columns:

            series = self.df[column]

            rows.append(
                {
                    "column": column,
                    "dtype": str(
                        series.dtype
                    ),
                    "min": series.min(),
                    "max": series.max(),
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                    "q01": series.quantile(0.01),
                    "q25": series.quantile(0.25),
                    "q75": series.quantile(0.75),
                    "q95": series.quantile(0.95),
                    "q99": series.quantile(0.99),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Transaction type profile
    # ------------------------------------------------------------------

    def transaction_type_profile(
        self,
    ) -> pd.DataFrame:

        if "type" not in self.df.columns:

            return pd.DataFrame()

        grouped = (
            self.df
            .groupby(
                "type",
                observed=True,
            )
            .agg(
                transactions=(
                    "type",
                    "size",
                ),
                total_amount=(
                    "amount",
                    "sum",
                ),
                average_amount=(
                    "amount",
                    "mean",
                ),
                median_amount=(
                    "amount",
                    "median",
                ),
            )
            .reset_index()
        )

        grouped["transaction_percentage"] = (
            grouped["transactions"]
            / len(self.df)
            * 100
        )

        if "isFraud" in self.df.columns:

            fraud_counts = (
                self.df
                .groupby(
                    "type",
                    observed=True,
                )["isFraud"]
                .sum()
                .rename(
                    "fraud_transactions"
                )
            )

            grouped = grouped.merge(
                fraud_counts,
                on="type",
                how="left",
            )

            grouped["fraud_rate"] = (
                grouped["fraud_transactions"]
                / grouped["transactions"]
                * 100
            )

        return grouped

    # ------------------------------------------------------------------
    # Fraud profile
    # ------------------------------------------------------------------

    def fraud_profile(
        self,
    ) -> pd.DataFrame:

        if "isFraud" not in self.df.columns:

            return pd.DataFrame()

        total = len(self.df)

        fraud = int(
            self.df["isFraud"]
            .eq(1)
            .sum()
        )

        legitimate = (
            total
            - fraud
        )

        fraud_rate = (
            fraud
            / total
            * 100
        )

        return pd.DataFrame(
            [
                {
                    "metric": "total_transactions",
                    "value": total,
                },
                {
                    "metric": "fraud_transactions",
                    "value": fraud,
                },
                {
                    "metric": "legitimate_transactions",
                    "value": legitimate,
                },
                {
                    "metric": "fraud_rate_percentage",
                    "value": round(
                        fraud_rate,
                        6,
                    ),
                },
            ]
        )

    # ------------------------------------------------------------------
    # Temporal profile
    # ------------------------------------------------------------------

    def temporal_profile(
        self,
    ) -> pd.DataFrame:

        if "step" not in self.df.columns:

            return pd.DataFrame()

        grouped = (
            self.df
            .groupby("step")
            .agg(
                transactions=(
                    "step",
                    "size",
                ),
            )
            .reset_index()
        )

        if "isFraud" in self.df.columns:

            fraud_counts = (
                self.df
                .groupby("step")["isFraud"]
                .sum()
                .rename(
                    "fraud_transactions"
                )
            )

            grouped = grouped.merge(
                fraud_counts,
                on="step",
                how="left",
            )

            grouped["fraud_rate"] = (
                grouped["fraud_transactions"]
                / grouped["transactions"]
                * 100
            )

        return grouped

    # ------------------------------------------------------------------
    # Entity profile
    # ------------------------------------------------------------------

    def entity_profile(
        self,
    ) -> pd.DataFrame:

        rows = []

        for column, entity_type in [
            ("nameOrig", "origin"),
            ("nameDest", "destination"),
        ]:

            if column not in self.df.columns:
                continue

            counts = (
                self.df[column]
                .value_counts()
            )

            rows.append(
                {
                    "entity_type": entity_type,
                    "column": column,
                    "unique_entities": int(
                        counts.size
                    ),
                    "total_transactions": len(
                        self.df
                    ),
                    "mean_transactions_per_entity": (
                        counts.mean()
                    ),
                    "median_transactions_per_entity": (
                        counts.median()
                    ),
                    "max_transactions_per_entity": (
                        counts.max()
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Outlier profile
    # ------------------------------------------------------------------

    def outlier_profile(
        self,
    ) -> pd.DataFrame:

        rows = []

        numeric_columns = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]

        for column in numeric_columns:

            if column not in self.df.columns:
                continue

            series = self.df[column]

            q1 = series.quantile(
                0.25
            )

            q3 = series.quantile(
                0.75
            )

            iqr = q3 - q1

            upper_bound = (
                q3
                + 1.5 * iqr
            )

            lower_bound = (
                q1
                - 1.5 * iqr
            )

            lower_count = int(
                (series < lower_bound)
                .sum()
            )

            upper_count = int(
                (series > upper_bound)
                .sum()
            )

            rows.append(
                {
                    "column": column,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "lower_outlier_count": lower_count,
                    "upper_outlier_count": upper_count,
                    "total_outlier_count": (
                        lower_count
                        + upper_count
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate(
        self,
    ) -> dict[str, pd.DataFrame]:

        return {
            "dataset_summary":
                self.dataset_summary(),

            "completeness":
                self.completeness_profile(),

            "cardinality":
                self.cardinality_profile(),

            "numeric_profile":
                self.numeric_profile(),

            "transaction_types":
                self.transaction_type_profile(),

            "fraud_profile":
                self.fraud_profile(),

            "temporal_profile":
                self.temporal_profile(),

            "entity_profile":
                self.entity_profile(),

            "outlier_profile":
                self.outlier_profile(),
        }


def save_quality_report(
    report: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Persist all quality report components as CSV files.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, dataframe in report.items():

        output_path = (
            output_dir
            / f"{name}.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )