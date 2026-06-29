# Causal AI Lift Measurement

End-to-end causal inference pipeline for estimating the incremental revenue and conversion lift of
advertising campaigns. The pipeline runs on Habu Clean Compute (PySpark) and produces a single-row
lift estimate with confidence intervals and segment-level CATE breakdowns.

---

## Primary Pipeline (PSM + Double-ML)

```
FeatureEngg (q82A)
  └─ AllFeatures
       └─► ML-PSM  ──────────► PSMMatchedFeatures
                                    └─► ML-ATE  ──► lift estimate (summary row)
```

**This is the primary pipeline for testing and production.**

| Step | Module | Input macro | Output macro | Purpose |
|---|---|---|---|---|
| q82A | `FeatureEngg` | raw exposure + transaction data | `AllFeatures` | Feature engineering — one row per addressLink |
| q85A | `ML-PSM` | `AllFeatures` | `PSMMatchedFeatures` | Propensity Score Matching — treated + matched controls |
| q86A | `ML-ATE` | `PSMMatchedFeatures` | lift summary row | Double-ML ATE / CATE estimation |

---

## Alternative Pipeline (Exact-Stratum Matching + Double-ML)

```
FeatureEngg (q82A)
  └─ AllFeatures
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

### FeatureEngg

**Path:** `FeatureEngg/`
**Wheel:** `causal_ai_feature_engg-<version>-py3-none-any.whl`

Computes all features from raw inputs. One row per `addressLink`. Outputs the `AllFeatures` table
consumed by both ML-PSM and FeatureEnggStratify.

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

`incremental_lift`, `lift_percent`, `avg_treatment_amount`, `expected_amount`,
`total_row_count`, `treated_count`, `outcome_r2`, `treatment_auc`,
`lift_pct_ci_lower`, `lift_pct_ci_upper`, `lift_ci_lower`, `lift_ci_upper`, `lift_p_value`,
`ate_base`, `cate_existing_buyer`, `cate_lapsed`, `cate_young`, `cate_senior`, `cate_high_income`,
`cate_existing_buyer_p_value`, `cate_lapsed_p_value`, `cate_young_p_value`,
`cate_senior_p_value`, `cate_high_income_p_value`

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
