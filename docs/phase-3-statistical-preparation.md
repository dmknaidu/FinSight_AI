# Phase 3 — Step 1: Statistical Dataset Preparation

## 1. Purpose

Prepare the Phase 1 canonical dataset for statistical analysis.

This step establishes the statistical-analysis contract without introducing
machine-learning features or modifying the canonical dataset.

---

## 2. Source

The canonical dataset is the sole analytical source:

data/processed/canonical/finsight_canonical.parquet

The raw dataset must not be used directly for Phase 3 analysis.

---

## 3. Target Variable

The primary statistical target is:

isFraud

Interpretation:

- 0 = legitimate transaction
- 1 = fraudulent transaction

The target is retained as ground truth.

No transformation is applied to the target.

---

## 4. Variable Classification

### Continuous financial variables

- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest

These variables will be evaluated using distributional statistics,
robust statistics, transformations where appropriate, and
fraud/non-fraud comparisons.

### Temporal variable

- step

The step variable represents transaction time within the PaySim simulation.

Temporal analysis will evaluate transaction volume, fraud occurrence,
and temporal variation.

### Categorical variable

- type

Allowed transaction types:

- CASH_IN
- CASH_OUT
- DEBIT
- PAYMENT
- TRANSFER

### Entity variables

- nameOrig
- nameDest

These variables represent transaction entities.

They will not be treated as ordinary categorical predictors because
of their high cardinality.

Entity-level aggregation and behavioral analysis belong to later analysis
and feature-engineering stages.

### Binary variables

- isFraud
- isFlaggedFraud

isFraud is the primary target.

isFlaggedFraud is treated as an existing system indicator and will be
analyzed separately.

---

## 5. Statistical Population

The complete canonical dataset contains:

- 6,362,620 transactions
- 8,213 fraudulent transactions
- 6,354,407 legitimate transactions

Fraud prevalence is approximately:

0.129082%

The extreme class imbalance must be explicitly considered during
statistical analysis.

---

## 6. Primary Comparison Groups

The principal comparison is:

Fraudulent transactions
vs.
Legitimate transactions

where:

isFraud = 1

and

isFraud = 0

These groups will be compared across:

- transaction amount
- transaction type
- temporal behavior
- balance behavior
- entity behavior
- system fraud flags

---

## 7. Statistical Preparation Rules

The preparation stage must:

1. Preserve the canonical dataset.
2. Avoid modifying source values.
3. Preserve the fraud target.
4. Preserve transaction categories.
5. Preserve entity identifiers.
6. Identify variable types.
7. Detect missing values.
8. Detect infinite values.
9. Confirm valid target values.
10. Confirm valid categorical values.
11. Record class prevalence.
12. Record descriptive statistics required for downstream testing.

---

## 8. Missing Values

Missing values must be measured before statistical analysis.

No imputation will be performed at this stage.

If missing values are discovered, the appropriate treatment will be
determined based on the statistical analysis requiring the variable.

---

## 9. Infinite Values

Infinite values must be explicitly checked.

Infinite observations must not silently enter statistical tests.

---

## 10. Distribution Preparation

Continuous variables must be evaluated for:

- minimum
- maximum
- mean
- median
- standard deviation
- variance
- quartiles
- skewness
- kurtosis
- zero proportion

No transformation is applied automatically.

Transformations will only be introduced when justified by statistical
analysis.

---

## 11. Class Imbalance

Because fraud prevalence is approximately 0.129082%, statistical
analysis must avoid interpreting class frequency alone as evidence
of meaningful behavior.

Both statistical significance and effect magnitude must be considered.

---

## 12. Statistical Significance

Later statistical testing will consider:

- p-values
- confidence intervals
- effect sizes
- practical significance

A statistically significant result will not automatically be treated
as an important predictive signal.

---

## 13. Reproducibility

The preparation process must record:

- source path
- source row count
- source column count
- target variable
- variable classifications
- fraud count
- legitimate count
- fraud prevalence
- missing-value counts
- infinite-value counts
- transaction-type counts
- generation timestamp
- Git commit

---

## 14. Output

The preparation stage will generate metadata under:

reports/statistical/

No transformed analytical dataset is persisted at this stage.

The canonical dataset remains unchanged.

---

## 15. Downstream Contract

The outputs of this step will be consumed by:

Phase 3 — Statistical Testing
Phase 3 — Effect Size Analysis
Phase 3 — Association Analysis
Phase 3 — Multicollinearity Analysis
Phase 3 — Signal Evaluation

---

## 16. Design Principle

Statistical analysis must begin from an explicitly defined analytical
population and variable contract.

The objective is to distinguish:

Observed difference
        ↓
Statistical evidence
        ↓
Effect magnitude
        ↓
Practical significance
        ↓
Potential predictive signal