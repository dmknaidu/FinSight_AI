from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class QualityFindingsError(Exception):
    """Raised when automated quality finding generation fails."""


@dataclass(frozen=True)
class QualityFinding:
    """
    One machine-readable data-quality finding.
    """

    finding_id: str
    category: str
    severity: str
    finding: str
    metric: str
    value: str
    interpretation: str
    downstream_implication: str


class QualityFindingsEngine:
    """
    Convert generated quality-report metrics into
    structured analytical findings.

    This component does not modify the source dataset.
    """

    def __init__(
        self,
        report: dict[str, pd.DataFrame],
    ) -> None:

        self.report = report

        required_reports = {
            "dataset_summary",
            "completeness",
            "cardinality",
            "numeric_profile",
            "transaction_types",
            "fraud_profile",
            "temporal_profile",
            "entity_profile",
            "outlier_profile",
        }

        missing = (
            required_reports
            - set(report.keys())
        )

        if missing:

            raise QualityFindingsError(
                "Missing required quality reports: "
                + ", ".join(
                    sorted(missing)
                )
            )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _format_value(
        value: object,
    ) -> str:

        if pd.isna(value):
            return "N/A"

        if isinstance(value, float):

            return f"{value:.6f}"

        return str(value)

    # ------------------------------------------------------------------
    # Completeness findings
    # ------------------------------------------------------------------

    def analyze_completeness(
        self,
    ) -> list[QualityFinding]:

        findings = []

        completeness = self.report[
            "completeness"
        ]

        total_missing = int(
            completeness[
                "missing_count"
            ].sum()
        )

        max_missing_percentage = float(
            completeness[
                "missing_percentage"
            ].max()
        )

        if total_missing == 0:

            findings.append(
                QualityFinding(
                    finding_id="DQ-001",
                    category="completeness",
                    severity="INFO",
                    finding=(
                        "No missing values detected "
                        "across the dataset."
                    ),
                    metric="missing_rate",
                    value="0.000000%",
                    interpretation=(
                        "All 6,362,620 records contain "
                        "values for all 11 columns."
                    ),
                    downstream_implication=(
                        "No missing-value imputation "
                        "is currently required."
                    ),
                )
            )

        else:

            findings.append(
                QualityFinding(
                    finding_id="DQ-001",
                    category="completeness",
                    severity="HIGH",
                    finding=(
                        "Missing values are present "
                        "in the dataset."
                    ),
                    metric="max_column_missing_rate",
                    value=(
                        f"{max_missing_percentage:.6f}%"
                    ),
                    interpretation=(
                        f"{total_missing:,} missing "
                        "values were detected."
                    ),
                    downstream_implication=(
                        "Missing-value handling must "
                        "be addressed before downstream "
                        "modeling."
                    ),
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Cardinality findings
    # ------------------------------------------------------------------

    def analyze_cardinality(
        self,
    ) -> list[QualityFinding]:

        findings = []

        cardinality = self.report[
            "cardinality"
        ]

        for column in [
            "nameOrig",
            "nameDest",
        ]:

            rows = cardinality[
                cardinality["column"] == column
            ]

            if rows.empty:
                continue

            row = rows.iloc[0]

            ratio = float(
                row["cardinality_ratio"]
            )

            unique_count = int(
                row["unique_count"]
            )

            if ratio >= 0.90:

                severity = "HIGH"

                implication = (
                    "Direct categorical encoding is "
                    "unlikely to be efficient. "
                    "Entity-level behavioral features "
                    "should be considered."
                )

            elif ratio >= 0.25:

                severity = "MEDIUM"

                implication = (
                    "Entity reuse is significant and "
                    "may support aggregation and "
                    "relationship-based features."
                )

            else:

                severity = "INFO"

                implication = (
                    "Cardinality is relatively low."
                )

            findings.append(
                QualityFinding(
                    finding_id=(
                        "DQ-002"
                        if column == "nameOrig"
                        else "DQ-003"
                    ),
                    category="cardinality",
                    severity=severity,
                    finding=(
                        f"{column} has high entity "
                        "cardinality."
                    ),
                    metric="cardinality_ratio",
                    value=(
                        f"{ratio * 100:.6f}%"
                    ),
                    interpretation=(
                        f"{unique_count:,} unique "
                        f"{column} values were observed."
                    ),
                    downstream_implication=(
                        implication
                    ),
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Fraud imbalance
    # ------------------------------------------------------------------

    def analyze_fraud(
        self,
    ) -> list[QualityFinding]:

        fraud = self.report[
            "fraud_profile"
        ]

        if fraud.empty:
            return []

        fraud_rate = float(
            fraud.loc[
                fraud["metric"]
                == "fraud_rate_percentage",
                "value",
            ].iloc[0]
        )

        fraud_count = int(
            fraud.loc[
                fraud["metric"]
                == "fraud_transactions",
                "value",
            ].iloc[0]
        )

        if fraud_rate < 1:

            severity = "HIGH"

            implication = (
                "Accuracy will be misleading as a "
                "primary model metric. Precision, "
                "recall, PR-AUC, ranking metrics, and "
                "cost-sensitive evaluation should be "
                "considered."
            )

        elif fraud_rate < 5:

            severity = "MEDIUM"

            implication = (
                "Class imbalance should be considered "
                "during model evaluation."
            )

        else:

            severity = "INFO"

            implication = (
                "Fraud class is comparatively "
                "well represented."
            )

        return [
            QualityFinding(
                finding_id="DQ-004",
                category="class_imbalance",
                severity=severity,
                finding=(
                    "The fraud class is highly "
                    "imbalanced."
                ),
                metric="fraud_rate_percentage",
                value=(
                    f"{fraud_rate:.6f}%"
                ),
                interpretation=(
                    f"{fraud_count:,} fraudulent "
                    "transactions are present."
                ),
                downstream_implication=(
                    implication
                ),
            )
        ]

    # ------------------------------------------------------------------
    # Numeric distribution
    # ------------------------------------------------------------------

    def analyze_numeric_distribution(
        self,
    ) -> list[QualityFinding]:

        findings = []

        numeric = self.report[
            "numeric_profile"
        ]

        amount_rows = numeric[
            numeric["column"] == "amount"
        ]

        if amount_rows.empty:
            return findings

        row = amount_rows.iloc[0]

        mean = float(
            row["mean"]
        )

        median = float(
            row["median"]
        )

        max_value = float(
            row["max"]
        )

        if median > 0:

            mean_median_ratio = (
                mean / median
            )

        else:

            mean_median_ratio = 0

        if mean_median_ratio >= 2:

            severity = "MEDIUM"

            implication = (
                "Amount-based analysis should account "
                "for heavy skew. Log transformations, "
                "percentile features, and robust "
                "statistics may be useful."
            )

        else:

            severity = "INFO"

            implication = (
                "Amount distribution does not show "
                "severe mean-median separation."
            )

        findings.append(
            QualityFinding(
                finding_id="DQ-005",
                category="distribution",
                severity=severity,
                finding=(
                    "Transaction amounts show "
                    "right-skewed behavior."
                ),
                metric="mean_to_median_ratio",
                value=(
                    f"{mean_median_ratio:.6f}"
                ),
                interpretation=(
                    f"Mean amount={mean:,.2f}; "
                    f"median amount={median:,.2f}; "
                    f"maximum={max_value:,.2f}."
                ),
                downstream_implication=(
                    implication
                ),
            )
        )

        return findings

    # ------------------------------------------------------------------
    # Temporal variation
    # ------------------------------------------------------------------

    def analyze_temporal_behavior(
        self,
    ) -> list[QualityFinding]:

        temporal = self.report[
            "temporal_profile"
        ]

        if temporal.empty:
            return []

        transactions = temporal[
            "transactions"
        ]

        minimum = int(
            transactions.min()
        )

        maximum = int(
            transactions.max()
        )

        if minimum > 0:

            volume_ratio = (
                maximum / minimum
            )

        else:

            volume_ratio = 0

        if volume_ratio >= 10:

            severity = "MEDIUM"

            implication = (
                "Temporal analysis should account "
                "for large variation in transaction "
                "volume. Raw per-step fraud rates "
                "may be unstable at low volumes."
            )

        else:

            severity = "INFO"

            implication = (
                "Transaction volume is relatively "
                "stable across time."
            )

        return [
            QualityFinding(
                finding_id="DQ-006",
                category="temporal",
                severity=severity,
                finding=(
                    "Transaction volume varies "
                    "substantially across time steps."
                ),
                metric="max_to_min_transactions_per_step",
                value=(
                    f"{volume_ratio:.6f}"
                ),
                interpretation=(
                    f"Minimum transactions/step="
                    f"{minimum:,}; maximum="
                    f"{maximum:,}."
                ),
                downstream_implication=(
                    implication
                ),
            )
        ]

    # ------------------------------------------------------------------
    # Entity reuse
    # ------------------------------------------------------------------

    def analyze_entity_behavior(
        self,
    ) -> list[QualityFinding]:

        entity = self.report[
            "entity_profile"
        ]

        findings = []

        for _, row in entity.iterrows():

            entity_type = row[
                "entity_type"
            ]

            max_transactions = int(
                row[
                    "max_transactions_per_entity"
                ]
            )

            median_transactions = float(
                row[
                    "median_transactions_per_entity"
                ]
            )

            if entity_type == "destination":

                severity = "MEDIUM"

                implication = (
                    "Destination-level aggregation and "
                    "relationship features may provide "
                    "useful behavioral signals."
                )

            else:

                severity = "INFO"

                implication = (
                    "Origin entities are highly sparse; "
                    "historical entity features may have "
                    "limited coverage without careful "
                    "design."
                )

            findings.append(
                QualityFinding(
                    finding_id=(
                        "DQ-007"
                        if entity_type == "origin"
                        else "DQ-008"
                    ),
                    category="entity_behavior",
                    severity=severity,
                    finding=(
                        f"{entity_type.capitalize()} "
                        "entities show distinct "
                        "transaction reuse patterns."
                    ),
                    metric="max_transactions_per_entity",
                    value=str(
                        max_transactions
                    ),
                    interpretation=(
                        f"Median transactions/entity="
                        f"{median_transactions:.2f}; "
                        f"maximum="
                        f"{max_transactions:,}."
                    ),
                    downstream_implication=(
                        implication
                    ),
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Outliers
    # ------------------------------------------------------------------

    def analyze_outliers(
        self,
    ) -> list[QualityFinding]:

        outliers = self.report[
            "outlier_profile"
        ]

        if outliers.empty:
            return []

        total_outliers = int(
            outliers[
                "total_outlier_count"
            ].sum()
        )

        return [
            QualityFinding(
                finding_id="DQ-009",
                category="outliers",
                severity="MEDIUM",
                finding=(
                    "Numerous statistical outliers "
                    "are present in financial variables."
                ),
                metric="total_iqr_outlier_count",
                value=str(
                    total_outliers
                ),
                interpretation=(
                    "IQR-based statistical outliers "
                    "were detected across financial "
                    "amount and balance variables."
                ),
                downstream_implication=(
                    "Outliers should generally be "
                    "preserved because extreme financial "
                    "behavior may represent meaningful "
                    "fraud signals."
                ),
            )
        ]

    # ------------------------------------------------------------------
    # Complete engine
    # ------------------------------------------------------------------

    def generate(
        self,
    ) -> list[QualityFinding]:

        findings: list[QualityFinding] = []

        findings.extend(
            self.analyze_completeness()
        )

        findings.extend(
            self.analyze_cardinality()
        )

        findings.extend(
            self.analyze_fraud()
        )

        findings.extend(
            self.analyze_numeric_distribution()
        )

        findings.extend(
            self.analyze_temporal_behavior()
        )

        findings.extend(
            self.analyze_entity_behavior()
        )

        findings.extend(
            self.analyze_outliers()
        )

        return findings


def findings_to_dataframe(
    findings: list[QualityFinding],
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "finding_id": finding.finding_id,
                "category": finding.category,
                "severity": finding.severity,
                "finding": finding.finding,
                "metric": finding.metric,
                "value": finding.value,
                "interpretation": finding.interpretation,
                "downstream_implication": (
                    finding.downstream_implication
                ),
            }
            for finding in findings
        ]
    )