from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.cleaning_policy import (
    CleaningRule,
    Treatment,
    get_rule,
)


@dataclass(frozen=True)
class PolicyEvaluation:
    """Result of evaluating one cleaning-policy rule."""

    issue: str
    treatment: str
    count: int
    message: str


class CleaningPolicyEvaluator:
    """
    Evaluate the FinSight AI cleaning policy against a DataFrame.

    This class only measures policy findings.

    It does NOT:
    - modify records
    - remove records
    - quarantine records
    - create flags inside the DataFrame
    """

    def __init__(
        self,
        expected_columns: list[str],
    ) -> None:

        self.expected_columns = expected_columns

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _result(
        self,
        issue: str,
        count: int,
    ) -> PolicyEvaluation:

        rule: CleaningRule = get_rule(issue)

        return PolicyEvaluation(
            issue=issue,
            treatment=rule.treatment.value,
            count=count,
            message=rule.reason,
        )

    # ------------------------------------------------------------------
    # Structural issues
    # ------------------------------------------------------------------

    def evaluate_missing_required_columns(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        missing = (
            set(self.expected_columns)
            - set(df.columns)
        )

        return self._result(
            "missing_required_column",
            len(missing),
        )

    def evaluate_unexpected_columns(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        unexpected = (
            set(df.columns)
            - set(self.expected_columns)
        )

        return self._result(
            "unexpected_column",
            len(unexpected),
        )

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------

    def evaluate_missing_values(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        count = int(
            df.isna().sum().sum()
        )

        return self._result(
            "missing_required_value",
            count,
        )

    # ------------------------------------------------------------------
    # Transaction domain
    # ------------------------------------------------------------------

    def evaluate_invalid_transaction_types(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        if "type" not in df.columns:
            return self._result(
                "invalid_transaction_type",
                0,
            )

        allowed = {
            "CASH_IN",
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER",
        }

        invalid_mask = ~df["type"].isin(
            allowed
        )

        count = int(
            invalid_mask.sum()
        )

        return self._result(
            "invalid_transaction_type",
            count,
        )

    def evaluate_negative_amounts(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        if "amount" not in df.columns:
            return self._result(
                "negative_amount",
                0,
            )

        count = int(
            (df["amount"] < 0).sum()
        )

        return self._result(
            "negative_amount",
            count,
        )

    def evaluate_negative_balances(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        balance_columns = [
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]

        count = 0

        for column in balance_columns:

            if column not in df.columns:
                continue

            count += int(
                (df[column] < 0).sum()
            )

        return self._result(
            "negative_balance",
            count,
        )

    def evaluate_invalid_steps(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        if "step" not in df.columns:
            return self._result(
                "invalid_step",
                0,
            )

        count = int(
            (df["step"] < 0).sum()
        )

        return self._result(
            "invalid_step",
            count,
        )

    # ------------------------------------------------------------------
    # Fraud indicators
    # ------------------------------------------------------------------

    def evaluate_invalid_is_fraud(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        if "isFraud" not in df.columns:
            return self._result(
                "invalid_is_fraud",
                0,
            )

        count = int(
            (~df["isFraud"].isin({0, 1}))
            .sum()
        )

        return self._result(
            "invalid_is_fraud",
            count,
        )

    def evaluate_invalid_is_flagged_fraud(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        if "isFlaggedFraud" not in df.columns:
            return self._result(
                "invalid_is_flagged_fraud",
                0,
            )

        count = int(
            (~df["isFlaggedFraud"].isin({0, 1}))
            .sum()
        )

        return self._result(
            "invalid_is_flagged_fraud",
            count,
        )

    # ------------------------------------------------------------------
    # Entity identifiers
    # ------------------------------------------------------------------

    def evaluate_empty_entity_identifiers(
        self,
        df: pd.DataFrame,
    ) -> list[PolicyEvaluation]:

        results: list[PolicyEvaluation] = []

        for column, issue in [
            (
                "nameOrig",
                "empty_origin_identifier",
            ),
            (
                "nameDest",
                "empty_destination_identifier",
            ),
        ]:

            if column not in df.columns:

                results.append(
                    self._result(
                        issue,
                        0,
                    )
                )

                continue

            values = (
                df[column]
                .astype("string")
            )

            count = int(
                values.isna().sum()
                + values.str.strip().eq("").sum()
            )

            results.append(
                self._result(
                    issue,
                    count,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Duplicates
    # ------------------------------------------------------------------

    def evaluate_duplicates(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        count = int(
            df.duplicated(
                keep=False
            ).sum()
        )

        return self._result(
            "exact_duplicate",
            count,
        )

    # ------------------------------------------------------------------
    # Fraud preservation
    # ------------------------------------------------------------------

    def evaluate_fraud_transactions(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        if "isFraud" not in df.columns:
            return self._result(
                "fraud_transaction",
                0,
            )

        count = int(
            (df["isFraud"] == 1).sum()
        )

        return self._result(
            "fraud_transaction",
            count,
        )

    def evaluate_flagged_fraud_transactions(
        self,
        df: pd.DataFrame,
    ) -> PolicyEvaluation:

        if "isFlaggedFraud" not in df.columns:
            return self._result(
                "flagged_fraud_transaction",
                0,
            )

        count = int(
            (df["isFlaggedFraud"] == 1).sum()
        )

        return self._result(
            "flagged_fraud_transaction",
            count,
        )

    # ------------------------------------------------------------------
    # Complete evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
    ) -> list[PolicyEvaluation]:

        results: list[PolicyEvaluation] = []

        results.append(
            self.evaluate_missing_required_columns(
                df
            )
        )

        results.append(
            self.evaluate_unexpected_columns(
                df
            )
        )

        results.append(
            self.evaluate_missing_values(
                df
            )
        )

        results.append(
            self.evaluate_invalid_transaction_types(
                df
            )
        )

        results.append(
            self.evaluate_negative_amounts(
                df
            )
        )

        results.append(
            self.evaluate_negative_balances(
                df
            )
        )

        results.append(
            self.evaluate_invalid_steps(
                df
            )
        )

        results.append(
            self.evaluate_invalid_is_fraud(
                df
            )
        )

        results.append(
            self.evaluate_invalid_is_flagged_fraud(
                df
            )
        )

        results.extend(
            self.evaluate_empty_entity_identifiers(
                df
            )
        )

        results.append(
            self.evaluate_duplicates(
                df
            )
        )

        results.append(
            self.evaluate_fraud_transactions(
                df
            )
        )

        results.append(
            self.evaluate_flagged_fraud_transactions(
                df
            )
        )

        return results


def build_policy_report(
    evaluations: list[PolicyEvaluation],
) -> pd.DataFrame:
    """
    Convert policy evaluation results into a DataFrame
    suitable for reporting.
    """

    return pd.DataFrame(
        [
            {
                "issue": evaluation.issue,
                "treatment": evaluation.treatment,
                "count": evaluation.count,
                "message": evaluation.message,
            }
            for evaluation in evaluations
        ]
    )


def calculate_policy_summary(
    evaluations: list[PolicyEvaluation],
) -> dict[str, int]:
    """
    Calculate aggregate counts by policy treatment.
    """

    summary = {
        Treatment.PIPELINE_FAILURE.value: 0,
        Treatment.QUARANTINE.value: 0,
        Treatment.PRESERVE_FLAG.value: 0,
        Treatment.PRESERVE.value: 0,
    }

    for evaluation in evaluations:

        if evaluation.count <= 0:
            continue

        summary[evaluation.treatment] += (
            evaluation.count
        )

    return summary