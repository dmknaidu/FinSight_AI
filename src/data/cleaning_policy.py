from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Treatment(str, Enum):
    """
    Allowed treatments for data-quality findings.
    """

    QUARANTINE = "QUARANTINE"
    PRESERVE_FLAG = "PRESERVE_FLAG"
    PRESERVE = "PRESERVE"
    PIPELINE_FAILURE = "PIPELINE_FAILURE"


@dataclass(frozen=True)
class CleaningRule:
    """
    Definition of one cleaning policy rule.
    """

    issue: str
    treatment: Treatment
    reason: str


CLEANING_RULES = [
    CleaningRule(
        issue="missing_required_column",
        treatment=Treatment.PIPELINE_FAILURE,
        reason=(
            "The dataset cannot satisfy the schema contract."
        ),
    ),
    CleaningRule(
        issue="unexpected_column",
        treatment=Treatment.PIPELINE_FAILURE,
        reason=(
            "Unexpected schema elements require investigation."
        ),
    ),
    CleaningRule(
        issue="missing_required_value",
        treatment=Treatment.QUARANTINE,
        reason=(
            "Required information is unavailable."
        ),
    ),
    CleaningRule(
        issue="invalid_transaction_type",
        treatment=Treatment.QUARANTINE,
        reason=(
            "Transaction type is outside the approved domain."
        ),
    ),
    CleaningRule(
        issue="negative_amount",
        treatment=Treatment.QUARANTINE,
        reason=(
            "Transaction amounts must be non-negative."
        ),
    ),
    CleaningRule(
        issue="negative_balance",
        treatment=Treatment.QUARANTINE,
        reason=(
            "Balance fields must be non-negative."
        ),
    ),
    CleaningRule(
        issue="invalid_step",
        treatment=Treatment.QUARANTINE,
        reason=(
            "The temporal step violates the data contract."
        ),
    ),
    CleaningRule(
        issue="invalid_is_fraud",
        treatment=Treatment.QUARANTINE,
        reason=(
            "The ground-truth fraud indicator must be binary."
        ),
    ),
    CleaningRule(
        issue="invalid_is_flagged_fraud",
        treatment=Treatment.QUARANTINE,
        reason=(
            "The source fraud flag must be binary."
        ),
    ),
    CleaningRule(
        issue="empty_origin_identifier",
        treatment=Treatment.QUARANTINE,
        reason=(
            "The originating entity cannot be identified."
        ),
    ),
    CleaningRule(
        issue="empty_destination_identifier",
        treatment=Treatment.QUARANTINE,
        reason=(
            "The destination entity cannot be identified."
        ),
    ),
    CleaningRule(
        issue="exact_duplicate",
        treatment=Treatment.PRESERVE_FLAG,
        reason=(
            "Duplicates require investigation and should "
            "not be silently deleted."
        ),
    ),
    CleaningRule(
        issue="extreme_transaction_amount",
        treatment=Treatment.PRESERVE_FLAG,
        reason=(
            "Extreme financial values may represent "
            "fraud or legitimate high-value activity."
        ),
    ),
    CleaningRule(
        issue="extreme_balance",
        treatment=Treatment.PRESERVE_FLAG,
        reason=(
            "Extreme balances may contain useful behavioral "
            "risk information."
        ),
    ),
    CleaningRule(
        issue="balance_inconsistency",
        treatment=Treatment.PRESERVE_FLAG,
        reason=(
            "Balance behavior is transaction-type dependent "
            "and requires separate investigation."
        ),
    ),
    CleaningRule(
        issue="fraud_transaction",
        treatment=Treatment.PRESERVE,
        reason=(
            "Fraud-labelled transactions are valid target "
            "observations and must be preserved."
        ),
    ),
    CleaningRule(
        issue="rare_transaction_type",
        treatment=Treatment.PRESERVE,
        reason=(
            "Rarity does not imply invalidity."
        ),
    ),
    CleaningRule(
        issue="flagged_fraud_transaction",
        treatment=Treatment.PRESERVE,
        reason=(
            "The source fraud flag is a useful risk signal."
        ),
    ),
]


def get_cleaning_rules() -> list[CleaningRule]:
    """
    Return the complete cleaning policy.
    """

    return list(CLEANING_RULES)


def get_rule(
    issue: str,
) -> CleaningRule:
    """
    Retrieve the policy for a specific issue.
    """

    for rule in CLEANING_RULES:

        if rule.issue == issue:
            return rule

    raise KeyError(
        f"No cleaning rule defined for: {issue}"
    )