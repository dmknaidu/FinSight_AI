# Phase 3 — Step 6: Categorical Association Analysis

## 1. Purpose

Analyze the association between transaction type and fraud status beyond
the global chi-square hypothesis test.

The analysis identifies which transaction categories contribute most to
the observed association.

No source data is modified.

---

## 2. Source

The analysis uses:

data/processed/canonical/finsight_canonical.parquet

The canonical dataset remains immutable.

---

## 3. Variables

Categorical variable:

- type

Binary target:

- isFraud

---

## 4. Global Association Test

The primary global test is Pearson's chi-square test of independence.

### Null hypothesis

H0: Transaction type and fraud status are independent.

### Alternative hypothesis

H1: Transaction type and fraud status are associated.

Significance level:

alpha = 0.05

---

## 5. Effect Size

Cramér's V is used to quantify the overall association magnitude.

Statistical significance is interpreted separately from effect magnitude.

---

## 6. Row-Level Fraud Rates

For each transaction type:

fraud_rate =
fraud_transactions / total_transactions

This describes the prevalence of fraud within each category.

---

## 7. Fraud Composition

For each transaction type:

fraud_composition =
fraud_transactions / total_fraud_transactions

This describes the proportion of all fraudulent transactions belonging
to each transaction type.

Fraud rate and fraud composition answer different questions and are both
reported.

---

## 8. Expected Frequencies

Expected counts are calculated under the independence assumption.

Expected count:

E_ij =
(row_total_i × column_total_j) / grand_total

The analysis verifies that expected frequencies satisfy the minimum
diagnostic requirement.

---

## 9. Standardized Pearson Residuals

Standardized Pearson residuals are calculated as:

(observed - expected) / sqrt(expected)

Positive residuals indicate more observations than expected under
independence.

Negative residuals indicate fewer observations than expected.

Large absolute residuals indicate categories that contribute strongly to
the global association.

---

## 10. Chi-Square Contributions

For each cell:

chi_square_contribution =
(observed - expected)^2 / expected

Cell contributions are summed to recover the global chi-square statistic.

This identifies where the categorical association originates.

---

## 11. Interpretation

The analysis distinguishes:

- global statistical significance
- overall association magnitude
- category-level fraud rates
- category contribution to chi-square

A category with many transactions can contribute strongly to chi-square
even when its fraud rate is relatively small.

---

## 12. Data Integrity

No rows are removed.

No observations are resampled.

No values are transformed.

No ML features are created.

The canonical dataset remains unchanged.

---

## 13. Outputs

Reports are saved under:

reports/statistical/categorical_association/

Expected outputs:

- categorical profile
- observed contingency table
- expected contingency table
- standardized residuals
- chi-square contributions
- association summary

---

## 14. Downstream Use

The results inform categorical feature engineering and fraud-signal
evaluation.

Association does not automatically imply predictive usefulness.