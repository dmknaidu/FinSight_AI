# FinSight AI — Phase 3 Step 8
# Entity-Level Behavioral Analysis

## 1. Purpose

This step investigates behavioral characteristics of transaction
entities using the immutable canonical dataset.

The analysis focuses on:

- originating entities (`nameOrig`)
- destination entities (`nameDest`)
- entity reuse
- transaction activity
- transaction amount behavior
- fraud association
- fraud concentration
- temporal activity span

This is a statistical/descriptive analysis step.

No machine-learning features are created.

---

## 2. Source Dataset

The analysis uses:

`data/processed/canonical/finsight_canonical.parquet`

The canonical dataset is treated as immutable.

No source or canonical records are modified.

---

## 3. Entity Definitions

### Origin Entity

`nameOrig`

Represents the originating entity associated with a transaction.

### Destination Entity

`nameDest`

Represents the destination entity associated with a transaction.

Origin and destination entities are analyzed independently.

---

## 4. Entity Behavioral Measures

For each entity the analysis calculates:

- transaction count
- total transaction amount
- average transaction amount
- median transaction amount
- maximum transaction amount
- fraud transaction count
- flagged-fraud transaction count
- first observed step
- last observed step
- temporal activity span
- entity-level fraud rate
- whether the entity has at least one fraud transaction
- whether the entity has at least one flagged-fraud transaction

---

## 5. Entity Reuse Analysis

Entity reuse is evaluated using minimum transaction thresholds:

- 1 transaction
- 2 transactions
- 3 transactions
- 5 transactions
- 10 transactions

The analysis reports:

- eligible entity count
- eligible transaction count
- percentage of entities meeting each threshold

These thresholds are descriptive and are not model-feature definitions.

---

## 6. Fraud Association Analysis

Entities are divided into:

### Fraud-associated entities

Entities with at least one transaction where:

`isFraud = 1`

### Non-fraud-associated entities

Entities with no transactions where:

`isFraud = 1`

Fraud association is evaluated under the minimum transaction thresholds.

The following measures are reported:

- entity counts
- eligible transaction counts
- fraud transaction counts
- entity-level fraud rate
- transaction-weighted fraud rate
- median transaction count
- Mann-Whitney U statistic
- p-value
- rank-biserial correlation
- statistical significance at alpha = 0.05

---

## 7. Why Transaction Thresholds Matter

An entity with one transaction and one fraud transaction would have an
entity-level fraud rate of 100%.

This does not imply that the entity has a stable 100% fraud propensity.

Therefore entity-level fraud association must be interpreted together
with transaction exposure.

A threshold-based analysis reduces the risk of overinterpreting
extremely sparse entity histories.

---

## 8. Fraud Concentration

Fraud concentration is measured among the:

- top 1% most active entities
- top 5% most active entities
- top 10% most active entities

Entities are ranked by transaction count.

For each group the analysis measures:

- fraud transaction count
- share of all fraud transactions
- fraud monetary amount
- share of total fraud monetary amount

Origin and destination entities are evaluated separately.

---

## 9. Behavioral Comparison

Entities associated with fraud are compared with entities without fraud
using:

- transaction count
- total amount
- average amount
- median amount
- maximum amount
- temporal activity span

Because the financial variables were previously shown to be strongly
skewed, the primary comparison uses the Mann-Whitney U test rather than
assuming normality.

Rank-biserial correlation is reported as the effect-size measure.

---

## 10. Statistical Interpretation

Statistical significance is evaluated at:

`alpha = 0.05`

A statistically significant result indicates evidence of a difference
between the analyzed entity groups.

It does not establish causation.

Large datasets can produce extremely small p-values for practically
small differences, so effect size and median differences must be
considered together with significance.

---

## 11. Origin vs Destination

The analysis explicitly compares origin and destination entity reuse.

Earlier data-quality analysis established that:

- `nameOrig` has extremely high cardinality
- `nameDest` has substantially greater entity reuse
- origin entities have very limited observed reuse
- destination entities exhibit substantially greater reuse

Step 8 quantifies these patterns rather than assuming that either side
is inherently more predictive.

---

## 12. Leakage Boundary

This step is descriptive statistical analysis.

The complete labeled canonical dataset may therefore be used to describe
historical entity behavior.

The resulting entity profiles must not automatically be used as
machine-learning features.

Future feature engineering must define:

- point-in-time availability
- historical observation windows
- label availability
- train/test separation
- prevention of future-information leakage

No such model feature construction is performed in Step 8.

---

## 13. Output Reports

The analysis produces:

`reports/statistical/entity_behavior/origin_entity_profile.csv`

`reports/statistical/entity_behavior/destination_entity_profile.csv`

`reports/statistical/entity_behavior/origin_reuse_distribution.csv`

`reports/statistical/entity_behavior/destination_reuse_distribution.csv`

`reports/statistical/entity_behavior/origin_fraud_association.csv`

`reports/statistical/entity_behavior/destination_fraud_association.csv`

`reports/statistical/entity_behavior/entity_fraud_concentration.csv`

`reports/statistical/entity_behavior/entity_behavior_summary.csv`

---

## 14. Data Integrity

The canonical dataset is not modified.

No rows are deleted.

No rows are added.

No canonical column types are changed.

No machine-learning features are persisted into the canonical dataset.

All outputs are analytical reports derived from the canonical dataset.

---

## 15. Execution

Run:

```bash
python3 scripts/analyze_entity_behavior.py