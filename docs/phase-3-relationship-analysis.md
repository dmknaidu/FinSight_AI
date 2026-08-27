# FinSight AI — Phase 3 Step 9
# Relationship & Network-Level Fraud Analysis

## 1. Objective

Analyze structural relationships between origin and destination entities
in the canonical transaction dataset.

This step investigates whether fraudulent transactions are concentrated
within particular origin-to-destination relationships or within highly
connected entities.

No machine-learning features are created in this step.

The canonical dataset is never modified.

---

## 2. Input Dataset

Canonical dataset:

`data/processed/canonical/finsight_canonical.parquet`

Required columns:

- `nameOrig`
- `nameDest`
- `amount`
- `isFraud`
- `isFlaggedFraud`

Only these columns are loaded for the relationship analysis.

---

## 3. Relationship Definition

A relationship is defined as a unique:

`nameOrig -> nameDest`

origin-to-destination pair.

For each unique relationship, the following statistics are calculated:

- transaction count
- total transaction amount
- average transaction amount
- median transaction amount
- maximum transaction amount
- fraud transaction count
- fraud amount
- flagged-fraud transaction count
- fraud rate
- fraud presence indicator
- flagged-fraud presence indicator

---

## 4. Memory Optimization

The dataset contains millions of high-cardinality entity identifiers.

Directly grouping repeatedly on string identifiers is avoided.

The implementation:

1. Reads only required columns.
2. Encodes origin identifiers into integer codes.
3. Encodes destination identifiers into integer codes.
4. Performs one primary origin-to-destination aggregation.
5. Reuses the relationship profile for subsequent analyses.
6. Restores human-readable entity identifiers only for the persisted profile.

This minimizes repeated processing of the transaction-level dataset.

---

## 5. Relationship Reuse Analysis

Relationship reuse is evaluated using minimum transaction thresholds:

- 1 transaction
- 2 transactions
- 3 transactions
- 5 transactions
- 10 transactions

For each threshold, the analysis records:

- eligible relationships
- eligible transactions
- percentage of relationships
- percentage of transactions

This determines whether fraud-related relationships are generally
one-time or repeatedly reused.

---

## 6. Origin Connectivity

For each origin entity, the analysis calculates:

- unique destinations
- relationship count
- transaction count
- total amount
- fraud transaction count
- fraud amount
- fraud rate

This provides a descriptive view of how broadly each origin interacts
with destination entities.

---

## 7. Destination Connectivity

For each destination entity, the analysis calculates:

- unique origins
- relationship count
- transaction count
- total amount
- fraud transaction count
- fraud amount
- fraud rate

Destination connectivity is particularly important because previous
analysis identified substantially greater destination-side entity reuse.

---

## 8. Fraud Relationship Comparison

Relationships containing at least one fraud transaction are compared
with relationships containing no fraud transactions.

Variables evaluated:

- transaction count
- total amount
- average amount
- median amount
- maximum amount

The comparison uses a two-sided Mann-Whitney U test.

Effect magnitude is represented using rank-biserial correlation.

Statistical significance is evaluated at:

`alpha = 0.05`

---

## 9. Fraud Concentration

Relationships are ranked by transaction count.

Fraud concentration is measured among:

- top 1% of relationships
- top 5% of relationships
- top 10% of relationships

For each group, the analysis measures:

- fraud transaction share
- fraud amount share

This identifies whether repeated transaction relationships account for a
disproportionate amount of fraudulent activity.

---

## 10. Statistical Interpretation

P-values must not be interpreted independently of effect size.

Because the dataset contains millions of observations, very small
distributional differences may become statistically significant.

Rank-biserial correlation should therefore be considered alongside
statistical significance.

The analysis is descriptive and inferential.

It does not establish causality.

---

## 11. Expected Outputs

Outputs are written to:

`reports/statistical/relationship_analysis/`

Expected files:

- `relationship_profile.csv`
- `relationship_reuse.csv`
- `origin_connectivity.csv`
- `destination_connectivity.csv`
- `fraud_relationship_summary.csv`
- `relationship_fraud_concentration.csv`
- `connectivity_comparison.csv`

---

## 12. Data Integrity

The canonical dataset is read-only.

No records are deleted.

No values are transformed in the canonical dataset.

No machine-learning features are persisted.

No train/test split is performed.

No model is trained.

---

## 13. Execution

Run:

```bash
python3 scripts/analyze_relationships.py