# FinSight AI — Phase 1 Cleaning Policy

## Purpose

This document defines how FinSight AI handles data-quality issues
during the data-engineering pipeline.

The objective is to distinguish between:

1. Invalid data that cannot safely enter the analytical dataset.
2. Unusual financial behavior that may represent a legitimate fraud
   or risk signal.

Fraud-related observations must not be removed merely because they
are unusual.

---

## Core Principle

FinSight AI follows the principle:

> Data-quality problems should be isolated, while potentially
> meaningful financial behavior should be preserved.

The cleaning process therefore uses three primary treatments:

- QUARANTINE
- PRESERVE + FLAG
- PRESERVE

No record is permanently deleted by the cleaning pipeline.

---

## Treatment Categories

### 1. QUARANTINE

A record is quarantined when it violates a structural or domain
constraint and cannot safely be used for downstream processing.

Examples:

- Missing required values
- Invalid transaction type
- Negative transaction amount
- Negative balance
- Invalid step value
- Invalid fraud indicator
- Invalid flagged-fraud indicator
- Missing or empty entity identifier

Quarantined records remain available for investigation.

---

### 2. PRESERVE + FLAG

A record remains in the analytical dataset but receives an
investigation or quality flag.

Examples:

- Exact duplicate
- Extreme transaction amount
- Extreme balance
- Balance reconciliation anomaly
- Unusual transaction behavior

These observations may contain valuable fraud or behavioral signals.

---

### 3. PRESERVE

Records that satisfy the data contract are retained.

Fraud-labelled transactions are explicitly preserved.

The `isFraud` field represents the ground-truth target and is not
a data-quality indicator.

---

## Issue Treatment Matrix

| Issue | Treatment |
|---|---|
| Missing required column | Pipeline failure |
| Unexpected column | Pipeline failure |
| Missing required value | Quarantine |
| Invalid transaction type | Quarantine |
| Negative amount | Quarantine |
| Negative balance | Quarantine |
| Invalid step | Quarantine |
| Invalid isFraud | Quarantine |
| Invalid isFlaggedFraud | Quarantine |
| Empty origin identifier | Quarantine |
| Empty destination identifier | Quarantine |
| Exact duplicate | Preserve + flag |
| Extreme amount | Preserve + flag |
| Extreme balance | Preserve + flag |
| Balance inconsistency | Preserve + investigate |
| Fraud-labelled transaction | Preserve |
| Rare transaction type | Preserve |
| isFlaggedFraud = 1 | Preserve |

---

## Duplicate Policy

Exact duplicate records are not automatically deleted.

Reasons include:

- Duplicate ingestion may represent a pipeline issue.
- Repeated financial events may require business investigation.
- Removing duplicates can alter the fraud distribution.
- The original records should remain traceable.

Duplicates will therefore be detected and flagged.

A later investigation phase may determine whether deduplication
is appropriate for a particular downstream dataset.

---

## Fraud Policy

Fraudulent transactions are not considered dirty data.

A transaction with:

`isFraud = 1`

is a valid observation of fraudulent behavior and is therefore
preserved.

Removing fraudulent observations would introduce target bias and
could reduce the model's ability to learn meaningful fraud patterns.

---

## Outlier Policy

Extreme numerical values are not automatically removed.

Financial fraud detection is inherently interested in unusual
behavior.

Therefore:

- Extreme amounts are preserved.
- Extreme balances are preserved.
- Statistical outliers are preserved.

Where appropriate, they will later be converted into analytical
features or investigation flags.

---

## Balance Policy

Balance inconsistencies are not automatically treated as invalid
records.

The PaySim transaction model contains different transaction types
with different balance behaviors.

Therefore balance reconciliation will be investigated separately
and evaluated by transaction type.

The cleaning pipeline will not impose a universal balance equation.

---

## Traceability

The cleaning process must maintain traceability between:

- Original record
- Cleaning decision
- Reason
- Processing timestamp
- Resulting dataset

No original record should be permanently deleted by the cleaning
pipeline.

---

## Design Principle

The FinSight AI data pipeline prioritizes:

1. Data integrity
2. Reproducibility
3. Traceability
4. Preservation of fraud signals
5. Explicit cleaning decisions
6. Explainability of transformations