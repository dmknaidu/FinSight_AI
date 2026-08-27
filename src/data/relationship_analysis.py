"""
FinSight AI — Relationship & Network-Level Fraud Analysis
=========================================================

This module performs relationship-level and network-level statistical analysis
without modifying the canonical dataset.

The analysis explicitly handles datasets where origin-destination relationships
are entirely unique. In that case, repeated-relationship aggregation is not
performed unnecessarily and the result schemas remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


@dataclass
class RelationshipAnalysisConfig:
    """Configuration for relationship analysis."""

    relationship_thresholds: tuple = (1, 2, 3, 5, 10)
    concentration_percentages: tuple = (1, 5, 10)
    alpha: float = 0.05


class RelationshipAnalysis:
    """
    Perform relationship and network-level fraud analysis.

    Parameters
    ----------
    dataframe:
        Canonical transaction dataframe.

    config:
        Optional analysis configuration.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: RelationshipAnalysisConfig | None = None,
    ) -> None:
        self.df = dataframe
        self.config = config or RelationshipAnalysisConfig()

        self._validate_columns()

    def _validate_columns(self) -> None:
        """Validate required canonical columns."""

        required_columns = {
            "nameOrig",
            "nameDest",
            "amount",
            "isFraud",
            "isFlaggedFraud",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                "Relationship analysis requires the following missing "
                f"columns: {sorted(missing)}"
            )

    # ------------------------------------------------------------------
    # Basic population statistics
    # ------------------------------------------------------------------

    def build_population_summary(self) -> pd.DataFrame:
        """Build analysis population summary."""

        total_transactions = len(self.df)
        total_fraud_transactions = int(self.df["isFraud"].sum())

        overall_fraud_rate = (
            total_fraud_transactions / total_transactions
            if total_transactions > 0
            else np.nan
        )

        unique_origins = int(self.df["nameOrig"].nunique())
        unique_destinations = int(self.df["nameDest"].nunique())

        records = [
            {
                "metric": "total_transactions",
                "value": total_transactions,
            },
            {
                "metric": "total_fraud_transactions",
                "value": total_fraud_transactions,
            },
            {
                "metric": "overall_fraud_rate",
                "value": overall_fraud_rate,
            },
            {
                "metric": "unique_origin_entities",
                "value": unique_origins,
            },
            {
                "metric": "unique_destination_entities",
                "value": unique_destinations,
            },
        ]

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Relationship uniqueness diagnostic
    # ------------------------------------------------------------------

    def run_relationship_uniqueness_diagnostic(
        self,
    ) -> Dict[str, Any]:
        """
        Determine whether origin-destination relationships are reused.

        Uses duplicated detection rather than immediately constructing a large
        groupby aggregation.
        """

        pair_index = pd.MultiIndex.from_frame(
            self.df[["nameOrig", "nameDest"]],
        )

        duplicated_mask = pair_index.duplicated(keep=False)

        total_transactions = len(self.df)

        unique_relationships = int(
            pd.MultiIndex.from_frame(
                self.df[["nameOrig", "nameDest"]]
            ).nunique()
        )

        repeated_relationship_transactions = int(
            duplicated_mask.sum()
        )

        repeated_relationships_detected = (
            repeated_relationship_transactions > 0
        )

        summary = pd.DataFrame(
            [
                {
                    "metric": "total_transactions",
                    "value": total_transactions,
                },
                {
                    "metric": "unique_relationships",
                    "value": unique_relationships,
                },
                {
                    "metric": "repeated_relationship_transactions",
                    "value": repeated_relationship_transactions,
                },
                {
                    "metric": "relationship_reuse_detected",
                    "value": repeated_relationships_detected,
                },
                {
                    "metric": "all_relationships_unique",
                    "value": (
                        unique_relationships == total_transactions
                    ),
                },
            ]
        )

        return {
            "summary": summary,
            "all_relationships_unique": (
                unique_relationships == total_transactions
            ),
            "unique_relationships": unique_relationships,
            "repeated_relationship_transactions": (
                repeated_relationship_transactions
            ),
            "duplicated_mask": duplicated_mask,
        }

    # ------------------------------------------------------------------
    # Relationship profile
    # ------------------------------------------------------------------

    def build_relationship_profile(
        self,
        uniqueness_diagnostic: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Build relationship-level profile.

        When every relationship is unique, transaction-level rows themselves
        represent the relationship profile.
        """

        all_relationships_unique = (
            uniqueness_diagnostic["all_relationships_unique"]
        )

        if all_relationships_unique:

            profile = self.df[
                [
                    "nameOrig",
                    "nameDest",
                    "amount",
                    "isFraud",
                    "isFlaggedFraud",
                ]
            ].copy()

            profile.insert(
                2,
                "transaction_count",
                1,
            )

            profile.rename(
                columns={
                    "amount": "total_amount",
                    "isFraud": "fraud_transaction_count",
                    "isFlaggedFraud": (
                        "flagged_fraud_transaction_count"
                    ),
                },
                inplace=True,
            )

            profile["average_amount"] = (
                profile["total_amount"]
            )

            profile["median_amount"] = (
                profile["total_amount"]
            )

            profile["maximum_amount"] = (
                profile["total_amount"]
            )

            profile["fraud_amount"] = np.where(
                profile["fraud_transaction_count"] == 1,
                profile["total_amount"],
                0.0,
            )

            profile["fraud_rate"] = (
                profile["fraud_transaction_count"]
                / profile["transaction_count"]
            )

            profile["has_fraud"] = (
                profile["fraud_transaction_count"] > 0
            ).astype(int)

            profile["has_flagged_fraud"] = (
                profile["flagged_fraud_transaction_count"] > 0
            ).astype(int)

            profile = profile[
                [
                    "nameOrig",
                    "nameDest",
                    "transaction_count",
                    "total_amount",
                    "average_amount",
                    "median_amount",
                    "maximum_amount",
                    "fraud_transaction_count",
                    "fraud_amount",
                    "flagged_fraud_transaction_count",
                    "fraud_rate",
                    "has_fraud",
                    "has_flagged_fraud",
                ]
            ]

            return profile

        aggregation = (
            self.df.groupby(
                ["nameOrig", "nameDest"],
                observed=True,
                sort=False,
            )
            .agg(
                transaction_count=(
                    "amount",
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
                maximum_amount=(
                    "amount",
                    "max",
                ),
                fraud_transaction_count=(
                    "isFraud",
                    "sum",
                ),
                flagged_fraud_transaction_count=(
                    "isFlaggedFraud",
                    "sum",
                ),
            )
            .reset_index()
        )

        fraud_amount = (
            self.df.assign(
                _fraud_amount=np.where(
                    self.df["isFraud"] == 1,
                    self.df["amount"],
                    0.0,
                )
            )
            .groupby(
                ["nameOrig", "nameDest"],
                observed=True,
                sort=False,
            )["_fraud_amount"]
            .sum()
            .reset_index(name="fraud_amount")
        )

        profile = aggregation.merge(
            fraud_amount,
            on=["nameOrig", "nameDest"],
            how="left",
        )

        profile["fraud_rate"] = (
            profile["fraud_transaction_count"]
            / profile["transaction_count"]
        )

        profile["has_fraud"] = (
            profile["fraud_transaction_count"] > 0
        ).astype(int)

        profile["has_flagged_fraud"] = (
            profile["flagged_fraud_transaction_count"] > 0
        ).astype(int)

        return profile

    # ------------------------------------------------------------------
    # Relationship reuse
    # ------------------------------------------------------------------

    def analyze_relationship_reuse(
        self,
        relationship_profile: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze repeated relationship exposure."""

        total_relationships = len(relationship_profile)
        total_transactions = int(
            relationship_profile["transaction_count"].sum()
        )

        records: List[Dict[str, Any]] = []

        for threshold in self.config.relationship_thresholds:

            eligible = relationship_profile[
                relationship_profile["transaction_count"]
                >= threshold
            ]

            eligible_relationships = len(eligible)

            eligible_transactions = int(
                eligible["transaction_count"].sum()
            )

            relationship_percentage = (
                eligible_relationships / total_relationships * 100
                if total_relationships > 0
                else np.nan
            )

            transaction_percentage = (
                eligible_transactions / total_transactions * 100
                if total_transactions > 0
                else np.nan
            )

            records.append(
                {
                    "minimum_transactions": threshold,
                    "eligible_relationships": (
                        eligible_relationships
                    ),
                    "eligible_transactions": (
                        eligible_transactions
                    ),
                    "relationship_percentage": (
                        relationship_percentage
                    ),
                    "transaction_percentage": (
                        transaction_percentage
                    ),
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Connectivity profiles
    # ------------------------------------------------------------------

    def build_origin_connectivity(self) -> pd.DataFrame:
        """Build origin-level network connectivity profile."""

        fraud_amount = np.where(
            self.df["isFraud"].to_numpy() == 1,
            self.df["amount"].to_numpy(),
            0.0,
        )

        working = self.df[
            [
                "nameOrig",
                "nameDest",
                "amount",
                "isFraud",
            ]
        ].copy()

        working["_fraud_amount"] = fraud_amount

        profile = (
            working.groupby(
                "nameOrig",
                observed=True,
                sort=False,
            )
            .agg(
                unique_destinations=(
                    "nameDest",
                    "nunique",
                ),
                relationship_count=(
                    "nameDest",
                    "size",
                ),
                transaction_count=(
                    "amount",
                    "size",
                ),
                total_amount=(
                    "amount",
                    "sum",
                ),
                fraud_transaction_count=(
                    "isFraud",
                    "sum",
                ),
                fraud_amount=(
                    "_fraud_amount",
                    "sum",
                ),
            )
            .reset_index()
        )

        profile["fraud_rate"] = (
            profile["fraud_transaction_count"]
            / profile["transaction_count"]
        )

        return profile

    def build_destination_connectivity(self) -> pd.DataFrame:
        """Build destination-level network connectivity profile."""

        fraud_amount = np.where(
            self.df["isFraud"].to_numpy() == 1,
            self.df["amount"].to_numpy(),
            0.0,
        )

        working = self.df[
            [
                "nameOrig",
                "nameDest",
                "amount",
                "isFraud",
            ]
        ].copy()

        working["_fraud_amount"] = fraud_amount

        profile = (
            working.groupby(
                "nameDest",
                observed=True,
                sort=False,
            )
            .agg(
                unique_origins=(
                    "nameOrig",
                    "nunique",
                ),
                relationship_count=(
                    "nameOrig",
                    "size",
                ),
                transaction_count=(
                    "amount",
                    "size",
                ),
                total_amount=(
                    "amount",
                    "sum",
                ),
                fraud_transaction_count=(
                    "isFraud",
                    "sum",
                ),
                fraud_amount=(
                    "_fraud_amount",
                    "sum",
                ),
            )
            .reset_index()
        )

        profile["fraud_rate"] = (
            profile["fraud_transaction_count"]
            / profile["transaction_count"]
        )

        return profile

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def _compare_groups(
        self,
        fraud_values: pd.Series,
        legitimate_values: pd.Series,
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney comparison safely."""

        fraud_values = fraud_values.dropna()
        legitimate_values = legitimate_values.dropna()

        if (
            len(fraud_values) == 0
            or len(legitimate_values) == 0
        ):
            return {
                "mann_whitney_u": np.nan,
                "p_value": np.nan,
                "rank_biserial_correlation": np.nan,
                "absolute_effect_size": np.nan,
                "significant_at_alpha_0_05": False,
            }

        result = mannwhitneyu(
            fraud_values,
            legitimate_values,
            alternative="two-sided",
        )

        n1 = len(fraud_values)
        n2 = len(legitimate_values)

        rank_biserial = (
            (2 * result.statistic) / (n1 * n2)
        ) - 1

        return {
            "mann_whitney_u": float(
                result.statistic
            ),
            "p_value": float(
                result.pvalue
            ),
            "rank_biserial_correlation": (
                rank_biserial
            ),
            "absolute_effect_size": abs(
                rank_biserial
            ),
            "significant_at_alpha_0_05": (
                result.pvalue < self.config.alpha
            ),
        }

    # ------------------------------------------------------------------
    # Fraud-associated relationships
    # ------------------------------------------------------------------

    def analyze_fraud_relationships(
        self,
        relationship_profile: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare fraud-associated and legitimate relationships."""

        fraud_relationships = relationship_profile[
            relationship_profile["has_fraud"] == 1
        ]

        legitimate_relationships = relationship_profile[
            relationship_profile["has_fraud"] == 0
        ]

        variables = [
            "transaction_count",
            "total_amount",
            "average_amount",
            "median_amount",
            "maximum_amount",
        ]

        records: List[Dict[str, Any]] = []

        for variable in variables:

            fraud_values = fraud_relationships[
                variable
            ]

            legitimate_values = legitimate_relationships[
                variable
            ]

            comparison = self._compare_groups(
                fraud_values,
                legitimate_values,
            )

            records.append(
                {
                    "variable": variable,
                    "fraud_relationship_count": (
                        len(fraud_relationships)
                    ),
                    "legitimate_relationship_count": (
                        len(legitimate_relationships)
                    ),
                    "fraud_relationship_median": (
                        fraud_values.median()
                        if len(fraud_values) > 0
                        else np.nan
                    ),
                    "legitimate_relationship_median": (
                        legitimate_values.median()
                        if len(legitimate_values) > 0
                        else np.nan
                    ),
                    "median_difference": (
                        fraud_values.median()
                        - legitimate_values.median()
                        if len(fraud_values) > 0
                        and len(legitimate_values) > 0
                        else np.nan
                    ),
                    **comparison,
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Concentration
    # ------------------------------------------------------------------

    def analyze_relationship_fraud_concentration(
        self,
        relationship_profile: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze fraud concentration across relationships.

        Ranking is based on transaction_count.
        """

        total_relationships = len(
            relationship_profile
        )

        total_fraud_transactions = int(
            relationship_profile[
                "fraud_transaction_count"
            ].sum()
        )

        total_fraud_amount = float(
            relationship_profile[
                "fraud_amount"
            ].sum()
        )

        ranked = relationship_profile.sort_values(
            "transaction_count",
            ascending=False,
            kind="stable",
        )

        records: List[Dict[str, Any]] = []

        for percentage in (
            self.config.concentration_percentages
        ):

            top_count = max(
                1,
                int(
                    np.ceil(
                        total_relationships
                        * percentage
                        / 100
                    )
                ),
            )

            top = ranked.head(
                top_count
            )

            top_fraud_transactions = int(
                top[
                    "fraud_transaction_count"
                ].sum()
            )

            top_fraud_amount = float(
                top[
                    "fraud_amount"
                ].sum()
            )

            records.append(
                {
                    "ranking_basis": (
                        "transaction_count"
                    ),
                    "top_relationship_percentage": (
                        percentage
                    ),
                    "top_relationship_count": (
                        top_count
                    ),
                    "total_relationships": (
                        total_relationships
                    ),
                    "total_fraud_transactions": (
                        total_fraud_transactions
                    ),
                    "top_relationships_fraud_transactions": (
                        top_fraud_transactions
                    ),
                    "fraud_transaction_share": (
                        top_fraud_transactions
                        / total_fraud_transactions
                        if total_fraud_transactions > 0
                        else np.nan
                    ),
                    "total_fraud_amount": (
                        total_fraud_amount
                    ),
                    "top_relationships_fraud_amount": (
                        top_fraud_amount
                    ),
                    "fraud_amount_share": (
                        top_fraud_amount
                        / total_fraud_amount
                        if total_fraud_amount > 0
                        else np.nan
                    ),
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Connectivity comparison
    # ------------------------------------------------------------------

    def analyze_connectivity_comparison(
        self,
        origin_connectivity: pd.DataFrame,
        destination_connectivity: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare connectivity of fraud-associated and legitimate entities.
        """

        records: List[Dict[str, Any]] = []

        analyses = [
            (
                "origin",
                origin_connectivity,
                "unique_destinations",
            ),
            (
                "destination",
                destination_connectivity,
                "unique_origins",
            ),
        ]

        for (
            entity_type,
            profile,
            connectivity_variable,
        ) in analyses:

            fraud_entities = profile[
                profile[
                    "fraud_transaction_count"
                ]
                > 0
            ]

            legitimate_entities = profile[
                profile[
                    "fraud_transaction_count"
                ]
                == 0
            ]

            fraud_values = fraud_entities[
                connectivity_variable
            ]

            legitimate_values = (
                legitimate_entities[
                    connectivity_variable
                ]
            )

            comparison = self._compare_groups(
                fraud_values,
                legitimate_values,
            )

            records.append(
                {
                    "entity_type": entity_type,
                    "variable": (
                        connectivity_variable
                    ),
                    "fraud_associated_entity_count": (
                        len(fraud_entities)
                    ),
                    "legitimate_entity_count": (
                        len(legitimate_entities)
                    ),
                    "fraud_associated_median": (
                        fraud_values.median()
                        if len(fraud_values) > 0
                        else np.nan
                    ),
                    "legitimate_median": (
                        legitimate_values.median()
                        if len(legitimate_values) > 0
                        else np.nan
                    ),
                    "median_difference": (
                        fraud_values.median()
                        - legitimate_values.median()
                        if len(fraud_values) > 0
                        and len(legitimate_values) > 0
                        else np.nan
                    ),
                    **comparison,
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete relationship and network-level analysis.
        """

        population_summary = (
            self.build_population_summary()
        )

        uniqueness = (
            self.run_relationship_uniqueness_diagnostic()
        )

        relationship_summary = (
            uniqueness["summary"]
        )

        relationship_profile = (
            self.build_relationship_profile(
                uniqueness
            )
        )

        relationship_reuse = (
            self.analyze_relationship_reuse(
                relationship_profile
            )
        )

        origin_connectivity = (
            self.build_origin_connectivity()
        )

        destination_connectivity = (
            self.build_destination_connectivity()
        )

        fraud_relationship_summary = (
            self.analyze_fraud_relationships(
                relationship_profile
            )
        )

        relationship_fraud_concentration = (
            self.analyze_relationship_fraud_concentration(
                relationship_profile
            )
        )

        connectivity_comparison = (
            self.analyze_connectivity_comparison(
                origin_connectivity,
                destination_connectivity,
            )
        )

        return {
            "population_summary": (
                population_summary
            ),
            "relationship_summary": (
                relationship_summary
            ),
            "relationship_profile": (
                relationship_profile
            ),
            "relationship_reuse": (
                relationship_reuse
            ),
            "origin_connectivity": (
                origin_connectivity
            ),
            "destination_connectivity": (
                destination_connectivity
            ),
            "fraud_relationship_summary": (
                fraud_relationship_summary
            ),
            "relationship_fraud_concentration": (
                relationship_fraud_concentration
            ),
            "connectivity_comparison": (
                connectivity_comparison
            ),
        }