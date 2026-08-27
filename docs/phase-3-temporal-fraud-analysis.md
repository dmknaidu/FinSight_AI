# Phase 3 — Step 7: Temporal Fraud Analysis

## 1. Purpose

Analyze temporal variation in transaction activity and fraud behavior using
the dataset's `step` variable.

The analysis distinguishes transaction volume, fraud count, fraud rate,
fraud monetary value, and fraud monetary share.

No source data is modified.

---

## 2. Source

The analysis uses:

data/processed/canonical/finsight_canonical.parquet

The canonical dataset remains immutable.

---

## 3. Temporal Variable

The temporal variable is:

- step

The dataset contains 743 observed time steps.

The analysis treats `step` as an ordered temporal index.

No calendar date or clock-time interpretation is inferred from the value
alone.

---

## 4. Per-Step Metrics

For every observed step, calculate:

- total transactions
- legitimate transactions
- fraud transactions
- fraud rate
- total transaction amount
- fraud amount
- legitimate amount
- fraud amount share
- average transaction amount
- average fraud amount

### Fraud rate

fraud_rate =
fraud_transactions / total_transactions

### Fraud amount share

fraud_amount_share =
fraud_amount / total_transaction_amount

---

## 5. Temporal Variability

Calculate:

- minimum transaction volume
- maximum transaction volume
- mean transaction volume
- standard deviation of transaction volume
- minimum fraud count
- maximum fraud count
- minimum fraud rate
- maximum fraud rate
- mean fraud rate
- standard deviation of fraud rate

The analysis identifies periods with unusually high or low activity.

---

## 6. Fraud Concentration

Calculate cumulative fraud concentration across ordered time steps.

Report:

- percentage of fraud occurring in the top 10% of time steps
- percentage of fraud amount occurring in the top 10% of time steps

The concentration analysis is based on observed fraud counts and monetary
amounts.

---

## 7. Volume vs Fraud Relationships

Calculate Pearson correlation between:

1. transaction volume and fraud count
2. transaction volume and fraud rate
3. transaction volume and fraud amount

Correlation is descriptive and does not establish causality.

---

## 8. Low-Volume Rate Stability

Fraud rates based on very small transaction counts can be unstable.

Therefore the analysis reports:

- transaction volume
- fraud count
- fraud rate

together.

No time steps are removed solely because of low volume.

---

## 9. Peak Period Identification

Identify:

- top time steps by transaction volume
- top time steps by fraud count
- top time steps by fraud rate
- top time steps by fraud amount

These rankings answer different questions and should not be conflated.

---

## 10. Data Integrity

No rows are removed.

No observations are resampled.

No values are transformed.

No ML features are created.

The canonical dataset remains unchanged.

---

## 11. Outputs

Reports are saved under:

reports/statistical/temporal/

Expected outputs:

- temporal profile
- temporal summary
- temporal correlations
- temporal concentration
- peak-period analysis

---

## 12. Downstream Use

The results inform:

- temporal feature engineering
- fraud-rate interpretation
- sequence and behavioral analysis
- time-aware model validation

Temporal findings do not establish causal relationships.

## 8A. Exposure-Aware Fraud Rates

Raw step-level fraud rates can become unstable when transaction volume is
small.

Therefore the analysis reports exposure-weighted fraud rates after applying
minimum transaction-volume thresholds of:

- 100 transactions
- 1,000 transactions
- 10,000 transactions

The exposure-weighted rate is calculated as:

eligible fraud transactions /
eligible transactions

The unweighted mean and median of eligible step-level rates are also
reported for descriptive comparison.

No observations are removed from the canonical dataset. Thresholds are
used only for analytical stability assessment.