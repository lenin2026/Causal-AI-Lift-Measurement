# Causal AI Lift Measurement

End-to-end causal inference pipeline for estimating the incremental revenue and conversion lift of
advertising campaigns. The pipeline runs on Habu Clean Compute (PySpark) and produces a single-row
lift estimate with confidence intervals and segment-level CATE breakdowns.

---

## Primary Pipeline (PSM + Double-ML)

```
q79A (Conversion Preprocessing)  ──┐
                                    ├──► FeatureEngg (q82A)
q80A (Exposure Preprocessing)  ────┘         └─ AllFeatures
                                                   └─► ML-PSM  ──────────► PSMMatchedFeatures
                                                                                └─► ML-ATE  ──► lift estimate (summary row)
```

**This is the primary pipeline for testing and production.**

| Step | Module | Input macro | Output macro | Purpose |
|---|---|---|---|---|
| q79A | `PreProcessing` | `@conversion`, `@mapping`, `@demographics`, `@campaign` | `raw_conversion` | Map conversion events to addressLink; flag campaign products |
| q80A | `PreProcessing` | `@exposure`, `@mapping`, `@demographics` | `raw_exposure` | Aggregate exposures to addressLink; assign treatment flag |
| q82A | `FeatureEngg` | `raw_conversion`, `raw_exposure`, `sample_insights` | `AllFeatures` | Feature engineering — one row per addressLink |
| q85A | `ML-PSM` | `AllFeatures` | `PSMMatchedFeatures` | Propensity Score Matching — treated + matched controls |
| q86A | `ML-ATE` | `PSMMatchedFeatures` | lift summary row | Double-ML ATE / CATE estimation |

---

## Alternative Pipeline (Exact-Stratum Matching + Double-ML)

```
q79A (Conversion Preprocessing)  ──┐
                                    ├──► FeatureEngg (q82A)
q80A (Exposure Preprocessing)  ────┘         └─ AllFeatures
                                                   └─► FeatureEnggStratify (q84B)  ──► StratifiedFeatures
                                                                                            └─► ML-ATE  ──► lift estimate
```

This pipeline uses exact-stratum 1:1 matching (poc_label × state_label × baseline_buyer_label ×
revenue_bin) rather than propensity score matching. Both estimators target the same causal
estimand; running both provides triangulation.

> **Note:** To switch ML-ATE between pipelines, change the input macro in
> `ML-ATE/client/transformation.py` from `PSMMatchedFeatures` to `StratifiedFeatures`.

---

## Module Reference

### PreProcessing

**Path:** `PreProcessing-PySpark-SQL-Queries/`

PySpark SQL queries that transform raw source tables into intermediate tables consumed by
FeatureEngg. No Python wheel — these run as SQL steps in Habu Clean Compute before q82A.

#### q79A — Conversion Preprocessing

**File:** `q79A_Conversion_Preprocessing.sql`
**Inputs:** `@conversion`, `@mapping`, `@demographics`, `@campaign`
**Output macro:** `raw_conversion`

Joins conversion events (`@conversion.lr_id`) through the identity graph (`@mapping` →
`@demographics`) to resolve each transaction to an `addressLink`. Deduplicates identity mappings
and demographic records by `install_date DESC`. Filters to the campaign window
(2024-11-12 – 2026-01-06) and deduplicates transactions by `(lr_id, order_id, timestamp,
product_id)`. Left-joins campaign product IDs to produce the `is_campaign_product` flag.

Key output columns: `addressLink`, `order_id`, `product_id`, `transaction_amount`, `quantity`,
`transaction_timestamp_unix`, `transaction_date`, `banner`, `division`, `transaction_category`,
`is_campaign_product`

#### q80A — Exposure Preprocessing

**File:** `q80A_Exposure_Preprocessing.sql`
**Inputs:** `@exposure`, `@mapping`, `@demographics`
**Output macro:** `raw_exposure`

Resolves Meta impression events (`@exposure.tp_id`) to `addressLink` via the identity graph.
Deduplicates exposures by `(tp_id, ts, campaign_id, ad_id, adset_id, account_id, event_type,
placement_type, device_platform, impression_device)`. Aggregates to the `addressLink` grain:
sets `treatment = 1` if any mapped online identity was exposed, computes exposure counts and
cardinality diagnostics, and flags `has_partial_exposure_within_addresslink` for households
with mixed exposed/unexposed identities.

Key output columns: `addressLink`, `treatment`, `is_eligible_control`, `min_exposure_ts`,
`exposure_frequency_deduped`, `mapped_online_identity_count`, `exposed_online_identity_count`,
`person_record_count`, `hhpel_count`, `has_partial_exposure_within_addresslink`

---

### FeatureEngg

**Path:** `FeatureEngg/`
**Wheel:** `causal_ai_feature_engg-<version>-py3-none-any.whl`

Computes all features from preprocessed inputs. One row per `addressLink`. Outputs the
`AllFeatures` table consumed by both ML-PSM and FeatureEnggStratify.

**Input macros:** `raw_conversion` (q79A output), `raw_exposure` (q80A output), `sample_insights`
(demographics, read directly without preprocessing)

Key outputs:

| Column | Description |
|---|---|
| `addressLink` | Household-level identifier |
| `treatment` | 1 = exposed, 0 = control candidate |
| `is_eligible_control` | 1 = no campaign-period exposure |
| `has_partial_exposure_within_addresslink` | 1 = mixed exposure within household |
| `baseline_12m_revenue_sum` | 12-month pre-campaign revenue |
| `baseline_12m_quantity_sum` | 12-month pre-campaign quantity |
| `baseline_12m_revenue_sum_bin` | Revenue stratum (8 tiers) used for exact matching |
| `baseline_buyer_label` | `recent_buyer` / `lapsed_buyer` / `no_12m_purchase` |
| `outcome_campaign_product_revenue` | Campaign-period revenue (outcome, not covariate) |
| `outcome_campaign_product_buyer` | Campaign-period buyer flag (outcome, not covariate) |

**Column rename history (v1.3):**

| Old name (pre-1.3) | New name (v1.3+) |
|---|---|
| `baseline_12m_revenue` | `baseline_12m_revenue_sum` |
| `baseline_12m_quantity` | `baseline_12m_quantity_sum` |
| `baseline_12m_revenue_bin` | `baseline_12m_revenue_sum_bin` |

Both ML-PSM and FeatureEnggStratify contain backward-compat rename blocks that handle old and new
names transparently.

---

### ML-PSM

**Path:** `ML-PSM/`
**Wheel:** `causal_ai_psm-<version>-py3-none-any.whl`
**Input:** `AllFeatures` (from FeatureEngg)
**Output:** `PSMMatchedFeatures` — individual rows for treated + PSM-matched controls

Implements Propensity Score Matching using Logistic Regression. The output is a row-level dataset
(not a summary) so that ML-ATE can run Double-ML on the PSM-balanced population.

**Matching pipeline:**

1. **Eligibility filter** — same as q84B: `treatment==1` OR `(treatment==0 AND is_eligible_control==1 AND has_partial_exposure==0)`
2. **Categorical encoding** — `StringIndexer(handleInvalid="keep")` on `poc_label`, `state_label`, `baseline_buyer_label`, `campaign_product_affinity_label`
3. **Control rebalancing** — downsample controls to `REBALANCE_RATIO=2 × treated_count` before fitting LR (prevents trivial prediction)
4. **Propensity score** — LR with `regParam=0.1`, `elasticNetParam=0.0` (L2), `maxIter=200`, `tol=1e-6`; scored on full eligible universe
5. **Clipping** — propensity scores clipped to `[0.05, 0.95]` to exclude non-overlap units
6. **Caliper bucketing** — bin width `CALIPER=0.01`; integer bin index avoids float equality issues
7. **1:1 matching** — within each bin, rank controls by `xxhash64(addressLink)` (deterministic); retain first `K = treated_in_bucket` controls per bin

**Output schema** — all original `AllFeatures` columns, plus:

| Column | Description |
|---|---|
| `propensity_score` | Clipped P(T=1\|X) from LR model |
| `treatment_group` | `"treatment"` or `"matched_control"` |

**Hyperparameter rationale:**

| Parameter | Value | Reason |
|---|---|---|
| `regParam` | 0.1 | Standard L2 penalty; matches reference PSM |
| `elasticNetParam` | 0.0 | Pure L2 — handles multicollinearity from 59+ correlated features better than L1 |
| `CALIPER` | 0.01 | Rosenbaum & Rubin (1985): ≈ 0.2 × SD of logit(PS) on probability scale |
| `PS_CLIP_LOW/HIGH` | 0.05 / 0.95 | Removes units without common support; same defensive pattern as ML-ATE |
| `REBALANCE_RATIO` | 2 | Keeps class imbalance manageable for LR; mirrors reference PSM |
| seed | 42 | Reproducibility |

**Data leakage:** See `ML-PSM/README.md` for the full exclusion table. Post-treatment outcome
columns (`outcome_campaign_product_*`) are excluded from propensity features. They are passed
through untouched to ML-ATE as outcome variables.

---

### FeatureEnggStratify

**Path:** `FeatureEnggStratify/`
**Wheel:** `causal_ai_feature_engg_stratification-<version>-py3-none-any.whl`
**Input:** `AllFeatures` (from FeatureEngg)
**Output:** `StratifiedFeatures` — individual rows for treated + exact-stratum-matched controls

Alternative matching strategy. Performs deterministic 1:1 exact-stratum matching on
`poc_label × state_label × baseline_buyer_label × baseline_12m_revenue_sum_bin`.

Within each stratum, controls are ranked by `xxhash64(addressLink)` and the first
`N = treated_count_in_stratum` controls are retained. Matching is **without replacement** —
`row_number()` never repeats a rank.

Use this pipeline when exact demographic and behavioral balance is more important than
propensity-score overlap.

---

### ML-ATE

**Path:** `ML-ATE/`
**Wheel:** `causal_ai_ate-<version>-py3-none-any.whl`
**Input:** `PSMMatchedFeatures` (primary) or `StratifiedFeatures` (alternative)
**Output:** Single-row lift summary

Implements Double-ML (cross-fitting / K-fold residualisation) for Average Treatment Effect (ATE)
and Conditional Average Treatment Effect (CATE) estimation.

**Model pipeline:**

1. **Column mapping** — `outcome_campaign_product_revenue` → `post_campaign_total_order_value`; `baseline_12m_revenue_sum` → `pre_campaign_total_order_value`
2. **Derived demographics** — `est_age` (weighted midpoint from age-bucket counts); `est_income_code` (weighted average of 35 income codes); majority `gender` from household counts
3. **99th-percentile winsorisation** — caps extreme spend values
4. **K=5 cross-fitting** — outcome model (Huber regression) and treatment model (LR) fitted on training folds, predictions generated on held-out folds
5. **Residualisation** — `y_res = Y - Ŷ`; `t_res = T - T̂`; ATE from OLS of `y_res ~ t_res`
6. **CATE** — interaction terms `t_res × segment` for 5 segments: existing buyer, lapsed, young, senior, high income

**Output columns (24):**

| # | Column | Spark Type | Source |
|---|---|---|---|
| 1 | `incremental_lift` | `DOUBLE` | `float(final_estimate.coefficients[0])` |
| 2 | `lift_percent` | `DOUBLE` | `float / float * 100` |
| 3 | `avg_treatment_amount` | `DOUBLE` | `F.avg(...)` — collected as Python `float` |
| 4 | `expected_amount` | `DOUBLE` | `float - float` |
| 5 | `total_row_count` | `LONG` | `F.count("*")` — collected as Python `int` |
| 6 | `treated_count` | `DOUBLE` | `F.sum(col.cast(DoubleType()))` — collected as Python `float` |
| 7 | `outcome_r2` | `DOUBLE` | `float(eval_y.evaluate(...))` |
| 8 | `treatment_auc` | `DOUBLE` | `float(eval_t.evaluate(...))` |
| 9 | `lift_pct_ci_lower` | `DOUBLE` | `float / float * 100` |
| 10 | `lift_pct_ci_upper` | `DOUBLE` | `float / float * 100` |
| 11 | `lift_ci_lower` | `DOUBLE` | `float - 1.96 * float` |
| 12 | `lift_ci_upper` | `DOUBLE` | `float + 1.96 * float` |
| 13 | `lift_p_value` | `DOUBLE` | `float(causal_summary.pValues[0])` |
| 14 | `ate_base` | `DOUBLE` | `float(coefs[0])` |
| 15 | `cate_existing_buyer` | `DOUBLE` | `float + float` |
| 16 | `cate_lapsed` | `DOUBLE` | `float + float` |
| 17 | `cate_young` | `DOUBLE` | `float + float` |
| 18 | `cate_senior` | `DOUBLE` | `float + float` |
| 19 | `cate_high_income` | `DOUBLE` | `float + float` |
| 20 | `cate_existing_buyer_p_value` | `DOUBLE` | `float(cate_summary.pValues[1])` or `-1.0` |
| 21 | `cate_lapsed_p_value` | `DOUBLE` | `float(cate_summary.pValues[2])` or `-1.0` |
| 22 | `cate_young_p_value` | `DOUBLE` | `float(cate_summary.pValues[3])` or `-1.0` |
| 23 | `cate_senior_p_value` | `DOUBLE` | `float(cate_summary.pValues[4])` or `-1.0` |
| 24 | `cate_high_income_p_value` | `DOUBLE` | `float(cate_summary.pValues[5])` or `-1.0` |

23 of 24 columns are `DOUBLE`. `total_row_count` is the only `LONG` — from `F.count("*")` which collects as a Python `int`.

**Switching between pipelines:**

In `ML-ATE/client/transformation.py`, change the read macro:

```python
# Primary pipeline (PSM → ATE)
psm_matched_features_df = self.data_handler.read("PSMMatchedFeatures")

# Alternative pipeline (exact-stratum → ATE)
stratified_features_df = self.data_handler.read("StratifiedFeatures")
```

---

## GCS Deployment (dev)

Bucket: `gs://habu-client-org-e22e5112-cd94-42bf-a2b9-6f95b52115c6/`
GCP project: `habu-client`

```bash
# FeatureEngg
cd FeatureEngg && python3 setup.py bdist_wheel
gsutil cp dist/causal_ai_feature_engg-<version>-py3-none-any.whl \
  gs://habu-client-org-e22e5112-cd94-42bf-a2b9-6f95b52115c6/

# ML-PSM
cd ML-PSM && python3 setup.py bdist_wheel
gsutil cp dist/causal_ai_psm-<version>-py3-none-any.whl \
  gs://habu-client-org-e22e5112-cd94-42bf-a2b9-6f95b52115c6/

# ML-ATE
cd ML-ATE && python3 setup.py bdist_wheel
gsutil cp dist/causal_ai_ate-<version>-py3-none-any.whl \
  gs://habu-client-org-e22e5112-cd94-42bf-a2b9-6f95b52115c6/

# FeatureEnggStratify (alternative pipeline only)
cd FeatureEnggStratify && python3 setup.py bdist_wheel
gsutil cp dist/causal_ai_feature_engg_stratification-<version>-py3-none-any.whl \
  gs://habu-client-org-e22e5112-cd94-42bf-a2b9-6f95b52115c6/
```

**Current wheels in GCS:**

| Wheel | Version |
|---|---|
| `causal_ai_feature_engg` | 1.2, 1.3 |
| `causal_ai_feature_engg_stratification` | 1.3 |
| `causal_ai_psm` | 1.3 |
| `causal_ai_ate` | 1.2, 1.3 |
| `clean_compute_spark_transformer` | 1.2 |

---

## Versioning

All modules follow the same `version/__init__.py` pattern:

```python
__version__ = "1.3"
```

Increment `__version__` before running `python3 setup.py bdist_wheel` to produce a new wheel.

---

## Key Design Decisions

### Why two matching strategies?

PSM (ML-PSM) and exact-stratum matching (FeatureEnggStratify) balance the treated/control
populations differently:

- **PSM** balances on the full covariate vector via a learned score — better when many continuous
  features matter and exact strata would be too sparse
- **Exact-stratum** guarantees perfect balance on the four stratum dimensions — better when those
  dimensions are known to be the dominant confounders

Running both and comparing lift estimates provides a consistency check: large disagreement
signals either poor PSM overlap or stratum sparsity.

### Why Double-ML (ML-ATE) rather than a simple mean comparison?

Simple matched-sample mean comparison (avg treated outcome − avg control outcome) is valid but
ignores residual covariate imbalance within propensity buckets. Double-ML residualises out both
the outcome and treatment propensity simultaneously, producing an estimate that is robust to
mis-specification of either nuisance model. The cross-fitting (K-fold) step eliminates in-sample
overfitting bias.

### Why is ML-ATE downstream of the matching step, not standalone?

Running ML-ATE on the unmatched AllFeatures universe is possible but less efficient — the
residualisation step has to do more work to account for the large treatment/control imbalance.
Pre-matching (either PSM or exact-stratum) reduces that imbalance before Double-ML, improving
the signal-to-noise ratio of the ATE estimate.

### Why not cross-join nearest-neighbour matching in PSM?

At campaign scale (millions of addressLinks) a full O(treated × control) cross-join would OOM.
Caliper bucketing reduces the join to within-bin pairs, which is O(treated × avg_bin_size).
This is equivalent to caliper matching for the purposes of balance while remaining scalable.
