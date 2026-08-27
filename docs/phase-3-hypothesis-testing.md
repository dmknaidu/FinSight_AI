# Phase 3 — Step 4: Hypothesis Testing

## 1. Purpose

Formally test predefined hypotheses concerning the relationship between
fraud status and transaction characteristics.

This step extends the descriptive and distributional findings from
previous statistical analysis.

No source data is modified.

---

## 2. Source

The analysis uses:

data/processed/canonical/finsight_canonical.parquet

The canonical dataset is immutable.

---

## 3. Hypothesis Family A — Transaction Amount

### Null hypothesis

H0: Transaction amount follows the same distribution for fraudulent and
legitimate transactions.

### Alternative hypothesis

H1: Transaction amount distributions differ between fraudulent and
legitimate transactions.

### Test

Two-sided Mann–Whitney U test.

---

## 4. Hypothesis Family B — Origin Balance

The origin balance variables are evaluated individually.

Variables:

- oldbalanceOrg
- newbalanceOrig

### Null hypothesis

H0: The distribution of the selected origin balance variable is the same
for fraudulent and legitimate transactions.

### Alternative hypothesis

H1: The distributions differ between fraudulent and legitimate transactions.

### Test

Two-sided Mann–Whitney U test.

---

## 5. Hypothesis Family C — Destination Balance

Variables:

- oldbalanceDest
- newbalanceDest

### Null hypothesis

H0: The distribution of the selected destination balance variable is the
same for fraudulent and legitimate transactions.

### Alternative hypothesis

H1: The distributions differ between fraudulent and legitimate transactions.

### Test

Two-sided Mann–Whitney U test.

---

## 6. Hypothesis Family D — Transaction Type

### Null hypothesis

H0: Transaction type and fraud status are statistically independent.

### Alternative hypothesis

H1: Transaction type and fraud status are associated.

### Test

Pearson chi-square test of independence.

### Effect size

Cramér's V.

---

## 7. Significance Level

The primary significance level is:

alpha = 0.05

For multiple continuous hypotheses, Benjamini–Hochberg false-discovery-rate
correction is applied.

---

## 8. Effect Size

Statistical significance is not treated as evidence of practical importance
by itself.

Continuous variables use rank-biserial correlation.

Transaction type uses Cramér's V.

---

## 9. Expected Frequencies

For the chi-square test, expected cell frequencies are calculated and
reported.

The test is considered diagnostically appropriate only when expected
frequencies are sufficiently large.

---

## 10. Interpretation

Each hypothesis receives:

- test statistic
- raw p-value
- adjusted p-value where applicable
- effect size
- significance decision
- interpretation

Results are interpreted using both statistical significance and practical
effect magnitude.

---

## 11. Multiple Testing

Five continuous hypotheses are evaluated together and corrected using
Benjamini–Hochberg FDR.

The transaction-type chi-square hypothesis is treated as a separate
categorical association test.

---

## 12. Data Integrity

No rows are removed.

No observations are resampled.

No values are transformed.

No ML features are created.

The canonical dataset remains unchanged.

---

## 13. Outputs

Reports are written to:

reports/statistical/hypothesis_testing/

Expected outputs:

- continuous hypothesis tests
- transaction-type contingency table
- categorical hypothesis test
- hypothesis-testing summary

---

## 14. Downstream Use

The results support:

- effect-size analysis
- feature signal evaluation
- feature engineering

Statistical significance alone does not determine feature selection.