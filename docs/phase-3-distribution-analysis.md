# Phase 3 — Step 2: Distribution & Normality Analysis

## 1. Purpose

Evaluate the statistical distributions of the continuous financial variables
before applying statistical tests.

This step determines whether conventional normality assumptions are reasonable
and establishes the appropriate statistical strategy for subsequent analysis.

No source values are transformed or modified.

---

## 2. Source

The analysis uses:

data/processed/canonical/finsight_canonical.parquet

The canonical dataset is treated as immutable.

---

## 3. Variables

The following continuous variables are analyzed:

- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest

---

## 4. Distribution Metrics

For each variable the analysis records:

- observation count
- missing count
- minimum
- maximum
- mean
- median
- standard deviation
- variance
- Q1
- Q3
- IQR
- skewness
- kurtosis
- zero count
- zero percentage
- IQR outlier count
- IQR outlier percentage

---

## 5. Normality Assessment

Because the dataset contains millions of observations, formal normality
tests can become excessively sensitive to very small deviations from a
normal distribution.

Therefore normality is assessed using both:

1. distributional statistics
2. a bounded statistical normality test sample

The normality test is diagnostic only.

A rejection of normality does not by itself determine whether a variable
is useful for statistical analysis.

---

## 6. Normality Testing Strategy

The complete dataset is not passed directly to a computationally expensive
normality test.

A deterministic sample is used for diagnostic testing.

The sample size is fixed and recorded in the output.

A random seed is fixed to ensure reproducibility.

---

## 7. Interpretation Rules

A variable will be considered approximately normal only when its distributional
characteristics and normality diagnostic provide reasonable support.

Strong skewness, substantial kurtosis, heavy tails, or large deviations
between mean and median will be treated as evidence against normality.

No automatic transformation is performed.

---

## 8. Outliers

IQR-based outliers are identified using:

Lower bound = Q1 - 1.5 × IQR

Upper bound = Q3 + 1.5 × IQR

Outliers are counted but not removed.

Extreme financial observations may represent legitimate or fraudulent
transaction behavior and therefore must be preserved.

---

## 9. Expected Statistical Strategy

If variables demonstrate strong non-normality, subsequent comparisons
will prefer robust or non-parametric methods where appropriate.

Possible downstream methods include:

- Mann–Whitney U
- rank-based effect measures
- bootstrap confidence intervals
- robust descriptive statistics

Method selection will be finalized in later steps based on the specific
research question.

---

## 10. Reproducibility

The analysis records:

- source dataset
- row count
- variables analyzed
- normality sample size
- random seed
- analysis timestamp
- Python environment
- Git commit

---

## 11. Outputs

Reports are written to:

reports/statistical/distributions/

The canonical dataset is not modified.

No ML features are created.

No statistical transformations are persisted.