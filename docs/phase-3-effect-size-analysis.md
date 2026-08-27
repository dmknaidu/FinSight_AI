# Phase 3 — Step 5: Effect Size Analysis

## 1. Purpose

Quantify the practical magnitude of differences between fraudulent and
legitimate transactions.

Statistical significance was established in previous hypothesis testing.
This step focuses on effect magnitude rather than additional significance
testing.

---

## 2. Source

The analysis uses:

data/processed/canonical/finsight_canonical.parquet

The canonical dataset remains immutable.

---

## 3. Variables

Effect sizes are calculated for:

- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest

---

## 4. Primary Effect Size

Rank-biserial correlation is used as the primary non-parametric effect-size
measure.

The orientation is:

positive  = fraud observations tend to have larger values

negative  = fraud observations tend to have smaller values

The theoretical range is:

-1 to +1

Values near zero indicate weak distributional separation.

Values farther from zero indicate stronger rank-based separation.

---

## 5. Common-Language Probability

The analysis reports the probability that a randomly selected fraudulent
observation has a greater value than a randomly selected legitimate
observation.

This probability is derived from the Mann–Whitney U statistic.

It provides an intuitive interpretation of rank-based separation.

---

## 6. Median Difference

For each variable:

median_difference =
fraud_median - legitimate_median

This provides a direct measure of the central-location difference.

---

## 7. Median Ratio

Where the legitimate median is non-zero:

median_ratio =
fraud_median / legitimate_median

When the legitimate median equals zero, the ratio is reported as undefined.

---

## 8. Practical Significance

Effect size is interpreted separately from statistical significance.

A very small effect may be statistically significant because of the large
sample size.

Therefore feature relevance will consider:

- effect magnitude
- direction
- distributional behavior
- domain interpretation
- downstream predictive usefulness

---

## 9. Interpretation Bands

For rank-biserial correlation, the following descriptive bands are used:

| Absolute effect | Interpretation |
|---|---|
| < 0.10 | negligible / very small |
| 0.10–<0.30 | small |
| 0.30–<0.50 | moderate |
| >= 0.50 | large |

These are interpretation guidelines rather than hard scientific thresholds.

---

## 10. Data Integrity

No rows are removed.

No observations are resampled.

No values are transformed.

No features are created.

The canonical dataset remains unchanged.

---

## 11. Outputs

Reports are saved under:

reports/statistical/effect_size/

Expected outputs:

- effect size profile
- effect size ranking

---

## 12. Downstream Use

The results inform:

Phase 3 — Feature Signal Evaluation
Phase 4 — Feature Engineering

Effect size does not automatically imply predictive performance.