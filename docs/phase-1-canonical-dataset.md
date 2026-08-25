# Phase 1 — Canonical Dataset Specification

## 1. Purpose

The canonical dataset is the validated and memory-optimized representation
of the source PaySim transaction dataset used by downstream FinSight AI
analytics, feature engineering, and machine learning pipelines.

The canonical dataset is derived from the raw dataset but does not replace
or modify the raw source.

---

## 2. Source Dataset

Source file:

PS_20174392719_1491204439457_log.csv

Expected source characteristics:

- Rows: 6,362,620
- Columns: 11
- Fraud transactions: 8,213
- Fraud rate: 0.129082%
- Missing values: 0
- Completely duplicated rows: 0

The raw source remains immutable.

---

## 3. Canonical Dataset Responsibilities

The canonical dataset must:

1. Preserve all valid source records.
2. Preserve the original column structure.
3. Preserve transaction values.
4. Preserve fraud labels.
5. Preserve entity identifiers.
6. Apply only approved dtype optimizations.
7. Exclude records only when explicitly required by the cleaning policy.
8. Preserve lineage to the source dataset.
9. Be suitable for repeated downstream consumption.

---

## 4. Approved Dtype Optimizations

The following optimizations were approved during Phase 1:

| Column | Source dtype | Canonical dtype |
|---|---|---|
| step | int64 | uint16 |
| type | object | category |
| isFraud | int64 | uint8 |
| isFlaggedFraud | int64 | uint8 |

Financial amount and balance columns remain `float64`.

The high-cardinality entity identifiers remain `object`.

---

## 5. Financial Columns

The following columns must remain `float64`:

- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest

No precision-reducing transformation is applied at canonicalization time.

---

## 6. Entity Identifiers

The following columns remain high-cardinality string/object columns:

- nameOrig
- nameDest

They must not be converted to categorical dtype during canonicalization.

Entity encoding and behavioral feature construction belong to later
feature-engineering stages.

---

## 7. Transaction Type

The `type` column is represented as a pandas categorical column.

Allowed values:

- CASH_IN
- CASH_OUT
- DEBIT
- PAYMENT
- TRANSFER

No transaction type is removed during canonicalization.

---

## 8. Fraud Labels

The following columns are preserved exactly:

- isFraud
- isFlaggedFraud

Fraud records must never be removed because they represent the target
population required for downstream fraud detection modeling.

---

## 9. Row Preservation

The canonicalization process must verify:

canonical rows + quarantine rows = source rows

For the current PaySim dataset:

- Source rows: 6,362,620
- Canonical rows: 6,362,620
- Quarantine rows: 0

Therefore the expected canonical dataset contains:

6,362,620 rows.

---

## 10. No Analytical Feature Engineering

Canonicalization must NOT introduce analytical features.

The canonical dataset must not contain fields such as:

- fraud_score
- risk_score
- transaction_velocity
- amount_percentile
- customer_risk
- destination_risk
- rolling_fraud_rate
- graph_features
- model_predictions

Those belong to later phases.

---

## 11. No Target Leakage

The canonicalization stage must not derive new features from the
fraud target.

The `isFraud` column is retained as ground truth, but it must not be
used to construct canonical features.

---

## 12. Persistence Format

The canonical dataset will be persisted in a columnar format suitable
for analytical workloads.

Parquet is the preferred format because it provides:

- columnar storage
- efficient compression
- selective column loading
- schema preservation
- better analytical performance than CSV
- compatibility with pandas and modern data-processing systems

The canonical dataset will therefore be stored as:

data/processed/canonical/

---

## 13. Canonical Dataset Metadata

The canonicalization pipeline must generate metadata containing:

- source dataset SHA-256
- canonical dataset SHA-256
- source row count
- canonical row count
- source column count
- canonical column count
- source schema fingerprint
- canonical schema fingerprint
- canonical dtypes
- canonical file size
- generation timestamp
- Git commit
- pipeline version

---

## 14. Validation Requirements

Before the canonical dataset is accepted, the pipeline must verify:

- source rows are accounted for
- canonical/quarantine partitions are mutually exclusive
- canonical columns match specification
- canonical dtypes match specification
- row count is correct
- fraud count is preserved
- flagged-fraud count is preserved
- no missing values are introduced
- no negative financial values are introduced
- transaction types remain valid
- source lineage is recorded

---

## 15. Downstream Contract

After Step 8 is complete, downstream FinSight AI components should
consume the canonical dataset rather than the raw CSV.

Raw data:

data/raw/

Canonical data:

data/processed/canonical/

The raw dataset remains the immutable source of truth.

The canonical dataset becomes the analytical source of truth.

---

## 16. Design Principle

The canonical dataset is not a "cleaned version" created for convenience.

It is a formally defined data product with:

- a schema
- validation rules
- lineage
- reproducibility metadata
- controlled transformations
- explicit downstream ownership