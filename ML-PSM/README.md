# ML-PSM: Propensity Score Matching Lift Model

## Purpose

This module estimates causal lift (incremental revenue effect of ad exposure) using Propensity Score
Matching (PSM). It receives the `AllFeatures` table produced by `FeatureEngg` (q82A step) directly as
its input — one row per `addressLink`, with treatment assignment, pre-campaign baseline behavior, and
campaign-period outcomes pre-computed.

PSM is used as a complementary estimator to the Double-ML / cross-fitting ATE in `ML-ATE`. Where
`ML-ATE` relies on residualisation, PSM reweights the control group so its covariate distribution
mirrors the treated group, then compares outcomes directly.

---

## Pipeline Position

```
FeatureEngg (q82A)
  └─ AllFeatures table  ──►  ML-PSM  ──►  lift estimate (summary DataFrame)
```

ML-PSM does **not** consume `FeatureEnggStratify` (q84B). q84B performs exact-stratum 1:1
sampling and is the input for `ML-ATE`. PSM does its own matching on propensity score and therefore
needs the full pre-matched universe from q82A.

---

## Data Leakage Analysis

All features used for propensity score estimation must be **pre-treatment** covariates only.

### Excluded from propensity features (would cause leakage)

| Column | Reason |
|---|---|
| `outcome_campaign_product_revenue` | Post-treatment outcome — the target variable |
| `outcome_campaign_product_orders` | Post-treatment outcome |
| `outcome_campaign_product_quantity` | Post-treatment outcome |
| `outcome_campaign_product_buyer` | Post-treatment outcome |
| `treatment` | The label being predicted by the propensity model |
| `is_eligible_control` | Treatment assignment machinery |
| `has_partial_exposure_within_addresslink` | Campaign-period exposure diagnostic |
| `min_exposure_ts` | Campaign-period timestamp |
| `exposure_frequency_deduped` | Campaign-period exposure count |
| `mapped_online_identity_count` | Identity graph diagnostic, not behavioral covariate |
| `exposed_online_identity_count` | Campaign-period diagnostic |
| `hhpel_count` | Identity graph diagnostic |
| `person_record_count` | Identity graph diagnostic |
| `online_identity_count` | Identity graph diagnostic |
| `hh_income_code_profile_label` | String label redundant with the 35 income-code count columns |
| `age_bucket_profile_label` | String label redundant with age bucket count columns |
| `baseline_12m_revenue_sum_bin` | Ordinal bin derived from `baseline_12m_revenue_sum` — redundant, causes multicollinearity |
| `addressLink` | Identifier |

### Not leakage — safe to use as propensity features

| Column | Why it is safe |
|---|---|
| `campaign_product_revenue_share` | Baseline window only: campaign-product revenue / total baseline revenue |
| `campaign_product_affinity_label` | Derived entirely from pre-campaign baseline purchase window |
| `prior_campaign_product_buyer` | Flag from baseline window only |

---

## Feature Set for Propensity Score Model

### Numeric (directly to VectorAssembler)

**12-month baseline behavior**
`baseline_12m_orders`, `baseline_12m_revenue_sum`, `baseline_12m_quantity_sum`,
`baseline_12m_negative_transaction_count`, `baseline_12m_distinct_banners`,
`baseline_12m_distinct_divisions`, `baseline_12m_active_purchase_days`,
`baseline_12m_avg_order_value`, `baseline_12m_avg_items_per_order`

**60-day baseline behavior**
`baseline_60d_orders`, `baseline_60d_revenue`, `baseline_60d_quantity`

**Campaign-product baseline**
`baseline_campaign_product_orders`, `baseline_campaign_product_revenue`,
`baseline_campaign_product_quantity`

**Recency and engagement**
`has_baseline_purchase`, `days_since_last_baseline_purchase`, `baseline_purchase_tenure_days`,
`num_weeks_purchased_in_last_365_days`, `recent_60d_revenue_share`, `recent_60d_order_share`,
`campaign_product_revenue_share`, `prior_campaign_product_buyer`, `recent_60d_buyer`, `lapsed_60d_buyer`

**Demographics — income codes (35)**
`num_hh_income_code_1_in_addresslink` … `num_hh_income_code_35_in_addresslink`

**Demographics — income edge cases (2)**
`num_hh_income_missing_in_addresslink`, `num_hh_income_zero_or_negative_in_addresslink`

**Demographics — age buckets (10)**
`age_missing_count`, `age_lt_18_count`, `age_18_24_count`, `age_25_34_count`, `age_35_44_count`,
`age_45_54_count`, `age_55_64_count`, `age_65_74_count`, `age_75_84_count`, `age_85_plus_count`

**Demographics — gender (3)**
`num_male_in_addresslink`, `num_female_in_addresslink`, `num_unknown_or_other_gender_in_addresslink`

### Categorical → StringIndexer → numeric index

`poc_label` → `poc_label_index`
`state_label` → `state_label_index`
`baseline_buyer_label` → `baseline_buyer_label_index`
`campaign_product_affinity_label` → `campaign_product_affinity_label_index`

`handleInvalid="keep"` on all StringIndexers — assigns unseen labels their own index rather than
failing, which matters when control units have state/poc combinations not seen in treatment.

---

## Implementation Plan

### Step 1 — Eligibility filter

Apply the same filter as q84B `match_ready`:
- Treatment: `treatment == 1`
- Eligible control: `treatment == 0 AND is_eligible_control == 1 AND has_partial_exposure == 0`

This ensures PSM and stratified matching operate on the same underlying universe.

### Step 2 — Categorical encoding

`StringIndexer(handleInvalid="keep")` on 4 categorical columns.

### Step 3 — Control group rebalancing

If `control_count > treated_count * REBALANCE_RATIO` (default 2), downsample control before fitting.
Matches reference PSM exactly:
- Prevents LR from trivially predicting "control"
- Seed = 42 for reproducibility

### Step 4 — Propensity score estimation

Logistic Regression using reference PSM hyperparameters:

```python
regParam=0.1, elasticNetParam=0   # L2 regularization
maxIter=200, tol=1e-6              # deterministic convergence (matching reference)
```

Score the full eligible universe (not just rebalanced sample) to get propensity scores for all units.
Propensity score = `probability[1]` via `VectorSlicer` + UDF — same as reference PSM.

### Step 5 — Nearest-neighbour caliper matching in PySpark

Reference PSM calls `knn_approximate_caliper`. This is implemented natively in PySpark:

1. Clip propensity scores to `[0.05, 0.95]` — removes units with near-zero overlap
2. Bucket scores into bins of width `CALIPER = 0.01` (100 bins): `floor(ps / caliper) * caliper`
3. Within each bin, rank treated and control separately by `xxhash64(addressLink)` (deterministic,
   consistent with q84B ordering)
4. Keep control where `stratum_rank <= treatment_count_in_bin` (1:K matching, default K=1)

This is equivalent to caliper matching and avoids a full O(treated × control) cross join.

**Why not exact cross-join nearest neighbour?**
At campaign scale (millions of addressLinks) a full cross join would OOM. Caliper bucketing reduces
the join to within-bin pairs only, which is O(treated × avg_bin_size).

### Step 6 — Outcome comparison

Compare `outcome_campaign_product_revenue` and `outcome_campaign_product_buyer` between treated and
matched control. Compute lift, incremental revenue, and conversion rate lift.

### Step 7 — Covariate balance (SMD)

Compute Standardised Mean Difference (SMD) post-match for key covariates:

```
SMD = (mean_treated - mean_control) / pooled_stddev
```

SMD < 0.1 is the conventional threshold (Austin 2011). Output as diagnostic columns in the result row.

---

## Output Schema

Single-row summary DataFrame:

| Column | Description |
|---|---|
| `treated_count` | Treated addressLinks after eligibility filter |
| `control_count_pre_match` | Eligible controls before matching |
| `matched_control_count` | Controls retained after PSM |
| `match_rate` | `matched_control_count / treated_count` |
| `propensity_model_auc` | LR AUC on full eligible universe |
| `avg_revenue_treated` | Mean `outcome_campaign_product_revenue` for treated |
| `avg_revenue_control` | Mean `outcome_campaign_product_revenue` for matched control |
| `incremental_revenue` | `(avg_revenue_treated - avg_revenue_control) * treated_count` |
| `lift_revenue_pct` | Revenue lift % |
| `conversion_rate_treated` | `avg(outcome_campaign_product_buyer)` for treated |
| `conversion_rate_control` | `avg(outcome_campaign_product_buyer)` for matched control |
| `lift_conversion_pct` | Conversion rate lift % |
| `smd_baseline_12m_revenue_sum` | SMD for revenue covariate post-match |
| `smd_baseline_12m_orders` | SMD for orders covariate post-match |
| `smd_days_since_last_baseline_purchase` | SMD for recency covariate post-match |

---

## Key Design Decisions

### Why FeatureEngg (q82A) output directly — not FeatureEnggStratify (q84B)?

q84B does exact-stratum 1:1 sampling. Feeding an already-matched sample into PSM would mean
matching on a matched sample — double selection, removing the very variation PSM needs to balance.
PSM must see the full eligible universe to choose its own matches.

### Why L2 regularization?

L2 (`elasticNetParam=0`) shrinks all coefficients smoothly without zeroing any. With 59+ correlated
features (income codes, age buckets), L2 handles multicollinearity better than L1 and produces
stable propensity scores. Matches reference PSM default.

### Why caliper = 0.01?

Rosenbaum & Rubin (1985) recommend caliper ≈ 0.2 × SD of logit(PS), which typically translates
to 0.01–0.02 on the probability scale. Tighter calipers improve balance but reduce match rate.

### Why clip to [0.05, 0.95]?

Units with extreme propensity scores (near 0 or 1) have little or no overlap between treatment and
control. Matching on these produces unstable estimates. Same defensive pattern as ML-ATE.

### Why SMD in output?

SMD is the standard PSM quality diagnostic (Austin 2011). An SMD > 0.1 post-match on any key
covariate indicates poor balance and the lift estimate is unreliable. Embedding it in the output
row makes it visible in results without separate tooling.

---

## Deploying to Google Cloud Storage (dev)

```bash
cd ML-PSM
python3 setup.py bdist_wheel
gsutil cp dist/causal_ai_psm-<version>-py3-none-any.whl \
  gs://habu-client-org-e22e5112-cd94-42bf-a2b9-6f95b52115c6/
```

GCS bucket: `gs://habu-client-org-e22e5112-cd94-42bf-a2b9-6f95b52115c6/`
GCP project: `habu-client`

---

## Testing

```bash
cd tests
python3 synthetic_validator.py
```
