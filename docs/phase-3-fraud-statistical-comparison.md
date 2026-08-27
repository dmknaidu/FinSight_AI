# Phase 3 — Step 3: Fraud/Non-Fraud Statistical Comparison

## 1. Purpose

Determine whether continuous transaction and balance variables exhibit
statistically meaningful differences between fraudulent and legitimate
transactions.

This step follows the distribution and normality analysis from Step 2.

No source data is modified.

---

## 2. Source

The analysis uses:

data/processed/canonical/finsight_canonical.parquet

The canonical dataset is immutable.

---

## 3. Comparison Groups

The primary comparison is:

Fraudulent transactions:

isFraud = 1

versus

Legitimate transactions:

isFraud = 0

---

## 4. Variables

The following continuous variables are compared:

- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest

---

## 5. Descriptive Analysis

For each variable and fraud group, the following are recorded:

- observation count
- mean
- median
- standard deviation
- Q1
- Q3
- minimum
- maximum

Median and quartile statistics receive particular attention because the
variables were shown to be strongly skewed in Step 2.

---

## 6. Statistical Test

The primary comparison uses the two-sided Mann–Whitney U test.

The test is selected because the continuous variables demonstrate strong
non-normality and substantial skewness.

The null hypothesis is:

H0: The fraud and legitimate groups have the same distribution.

The alternative hypothesis is:

H1: The fraud and legitimate groups have different distributions.

The significance threshold is:

alpha = 0.05

---

## 7. Effect Size

Statistical significance alone is insufficient because the dataset contains
millions of observations.

Rank-biserial correlation is reported as the primary effect-size measure.

Effect size is interpreted alongside the p-value rather than replacing it.

---

## 8. Multiple Testing

Five primary statistical tests are performed.

The Benjamini–Hochberg procedure is applied to control the false discovery
rate across these tests.

Both raw and adjusted p-values are reported.

---

## 9. Interpretation

A variable is not considered practically important solely because its
p-value is statistically significant.

Interpretation considers:

- adjusted p-value
- effect size
- median difference
- distributional separation
- domain relevance

---

## 10. Class Imbalance

Fraud represents approximately 0.129082% of transactions.

The large imbalance is explicitly documented.

The analysis does not artificially oversample or undersample the groups.

The observed population is retained.

---

## 11. Data Integrity

The following must remain unchanged:

- row count
- source values
- target values
- transaction types
- entity identifiers

No statistical transformation is persisted.

---

## 12. Outputs

Reports are saved under:

reports/statistical/fraud_comparison/

Expected outputs:

- group descriptive statistics
- Mann–Whitney test results
- statistical comparison summary

---

## 13. Downstream Use

The findings will inform:

Phase 3 — Effect Size Analysis
Phase 3 — Feature Signal Evaluation
Phase 4 — Feature Engineering

Statistical evidence does not automatically determine feature selection.