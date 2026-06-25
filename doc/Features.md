# Feature Documentation

## Pipeline Overview

Features are produced in two sequential steps:

| Step | Module | Description |
|------|--------|-------------|
| q82A | `FeatureEngg` | Computes all purchase behavior, demographic, and baseline features per `addressLink`. Left-joins exposure assignment → conversion history → demographics. |
| q84B | `FeatureEnggStratify` | Deterministic 1-to-1 stratified matching. Adds sampling metadata columns; passes all q82A features through unchanged. |

**Reference dates**

| Window | Start | End | Purpose |
|--------|-------|-----|---------|
| 12-month baseline | 2024-11-12 | 2025-11-11 | Pre-campaign purchase covariates |
| 60-day baseline | 2025-09-13 | 2025-11-11 | Recent purchase momentum sub-window |
| Campaign (outcome) | 2025-11-12 | 2026-01-06 | Outcome measurement only — not used as covariates |
| Recency reference | — | 2025-11-11 | Anchor for `days_since_last_baseline_purchase` |

---

## Column Reference

Columns are grouped by category. The **Use** column indicates whether the feature is included in model training.

| Value | Meaning |
|-------|---------|
| `model` | Included as a covariate in model training |
| `excluded` | Present in output but removed before model training via `MODEL_FEATURE_EXCLUDE_COLUMNS` or by convention |
| `outcome` | Target variable — never a covariate |

---

### Identifiers

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `addressLink` | String | excluded | Household-level analysis key. Trimmed and cast from input. |

---

### Treatment & Eligibility

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `treatment` | Integer | excluded | 1 if any mapped online identity under `addressLink` was exposed to the campaign; 0 otherwise. Sourced from q80A exposure assignment. |
| `is_eligible_control` | Integer | excluded | 1 if `addressLink` has zero campaign-period exposure and is eligible to serve as a matched control; 0 otherwise. |

---

### Stratification & Sampling Metadata *(q84B only)*

Added by `FeatureEnggStratify`. Not present in q82A output.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `treatment_group` | String | excluded | `"treatment"` if `treatment = 1`, else `"candidate_control"`. Derived in q84B. |
| `sampled_unit` | Integer | excluded | Literal 1 for every row retained after deterministic stratified sampling. |
| `stratum_rank` | Long | excluded | Row number within `(stratum_columns, treatment_group)` ordered by `xxhash64(addressLink), addressLink`. Deterministic across runs. |
| `treatment_addresslinks` | Long | excluded | Count of treated `addressLink`s in the exact stratum (all four stratum label values match). |
| `candidate_control_addresslinks` | Long | excluded | Count of eligible candidate control `addressLink`s in the exact stratum before 1:1 sampling. |

---

### Exposure Diagnostics

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `has_partial_exposure_within_addresslink` | Integer | excluded | 1 if the same `addressLink` contains both exposed and unexposed mapped online identities; 0 otherwise. |
| `min_exposure_ts` | Long | excluded | Earliest campaign exposure timestamp (Unix epoch) rolled up to `addressLink`. Null if no exposure. |
| `exposure_frequency_deduped` | Long | excluded | Deduplicated campaign exposure event count under `addressLink`. |
| `mapped_online_identity_count` | Long | excluded | Count of mapped online identity IDs under `addressLink`. |
| `exposed_online_identity_count` | Long | excluded | Count of mapped online identities under `addressLink` that received at least one campaign exposure. |
| `hhpel_count` | Long | excluded | Count of distinct `hhpel` values under `addressLink`. Diagnostic only. |
| `person_record_count` | Long | excluded | Count of distinct `Grouping_Indicator` person records under `addressLink`. |
| `online_identity_count` | Long | excluded | Assignment count of online identity IDs under `addressLink`. |

---

### Stratum Labels

Used as both matching keys in q84B stratification and model covariates.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `poc_label` | String | model | Child-presence label derived from latest demographics per `addressLink`. Values: `has_child`, `no_child`, `missing`. `has_child` if any record has `poc` in `{true, 1, yes, y}`; `no_child` if any record has `poc` in `{false, 0, no, n}`; `missing` otherwise. |
| `state_label` | String | model | State rollup from latest demographics. Values: a US state string, `mixed_or_multiple` (when `state_distinct_count > 1`), or `missing` (when no state data). |
| `baseline_buyer_label` | String | model | Recency-based buyer stratum. `recent_buyer` if `baseline_60d_orders > 0`; `lapsed_buyer` if `baseline_12m_orders > 0` and `baseline_60d_orders = 0`; `no_12m_purchase` otherwise. |
| `baseline_12m_revenue_bin` | String | model | Revenue stratum derived from `baseline_12m_revenue_sum`. Bins: `zero` (= 0), `lt_10` (< 10), `10_to_49` (10–49), `50_to_99` (50–99), `100_to_249` (100–249), `250_to_499` (250–499), `500_to_999` (500–999), `1000_plus` (≥ 1000). |

---

### Gender Distribution

Sourced from latest demographic record per online identity, then rolled up to `addressLink`.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `num_male_in_addresslink` | Long | model | Count of demographic records where `lower(trim(gender))` is in `{m, male}`. Default 0. |
| `num_female_in_addresslink` | Long | model | Count of demographic records where `lower(trim(gender))` is in `{f, female}`. Default 0. |
| `num_unknown_or_other_gender_in_addresslink` | Long | model | Count of demographic records where gender is null or not in `{m, male, f, female}`. Default 0. |

---

### Income Profile

Sourced from latest demographic record per online identity, rolled up to `addressLink`. All numeric income columns default to 0; the profile label defaults to `"missing"`.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `hh_income_code_profile_label` | String | model | Pipe-delimited summary of income code distribution, e.g. `code_3:2\|code_7:1`. Format: `code_N:count`. `missing` if no demographic records. |
| `num_hh_income_missing_in_addresslink` | Long | model | Count of records where `hh_income` cast to Integer is null. |
| `num_hh_income_zero_or_negative_in_addresslink` | Long | model | Count of records where `hh_income ≤ 0`. |
| `num_hh_income_code_1_in_addresslink` … `num_hh_income_code_35_in_addresslink` | Long | model | Count of records where `hh_income = N` for N in 1..35. One column per code. |
| `num_hh_income_code_other_in_addresslink` | Long | model | Count of records where `hh_income > 35`. |

---

### Age Distribution

Sourced from latest demographic record per online identity, rolled up to `addressLink`. All numeric age columns default to 0; the profile label defaults to `"missing"`.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `age_bucket_profile_label` | String | model | Pipe-delimited list of age buckets present under the `addressLink`, e.g. `25_34\|35_44\|55_64`. `missing` if no age data. |
| `age_missing_count` | Long | model | Count of records where `age` is null. |
| `age_lt_18_count` | Long | model | Count of records where `age < 18`. |
| `age_18_24_count` | Long | model | Count of records where `18 ≤ age ≤ 24`. |
| `age_25_34_count` | Long | model | Count of records where `25 ≤ age ≤ 34`. |
| `age_35_44_count` | Long | model | Count of records where `35 ≤ age ≤ 44`. |
| `age_45_54_count` | Long | model | Count of records where `45 ≤ age ≤ 54`. |
| `age_55_64_count` | Long | model | Count of records where `55 ≤ age ≤ 64`. |
| `age_65_74_count` | Long | model | Count of records where `65 ≤ age ≤ 74`. |
| `age_75_84_count` | Long | model | Count of records where `75 ≤ age ≤ 84`. |
| `age_85_plus_count` | Long | model | Count of records where `age ≥ 85`. |

---

### 12-Month Baseline Purchase Behavior

Window: **2024-11-12 through 2025-11-11** (ends the day before campaign start). All columns default to 0 / 0.0 for households with no transaction records.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `baseline_12m_orders` | Long | model | `COUNT(DISTINCT order_id)` where `transaction_date` is in the 12-month window. |
| `baseline_12m_revenue_sum` | Double | model | `SUM(transaction_amount)` for rows where `transaction_amount > 0` and date is in window. Negative amounts (returns) excluded. |
| `baseline_12m_quantity_sum` | Double | model | `SUM(quantity)` for rows in the 12-month window. |
| `baseline_12m_negative_transaction_count` | Long | model | Count of rows in the window where `transaction_amount < 0`. Proxy for return/refund activity. |
| `baseline_12m_distinct_banners` | Long | model | `COUNT(DISTINCT banner)` for rows in the 12-month window. 0 if `banner` column is absent. |
| `baseline_12m_distinct_divisions` | Long | model | `COUNT(DISTINCT division)` for rows in the 12-month window. 0 if `division` column is absent. |
| `baseline_12m_active_purchase_days` | Long | model | `COUNT(DISTINCT transaction_date)` in the 12-month window. Measures purchase frequency spread. |
| `num_weeks_purchased_in_last_365_days` | Long | model | `COUNT(DISTINCT date_trunc('week', transaction_date))` in the 12-month window. |
| `has_baseline_purchase` | Integer | model | 1 if any purchase exists in the 12-month window, else 0. Companion flag for `days_since_last_baseline_purchase` and `baseline_purchase_tenure_days`. |
| `days_since_last_baseline_purchase` | Long | model | `DATEDIFF('2025-11-11', MAX(transaction_date))` for rows in the window. **366 for non-buyers** (one day beyond the 12-month window maximum of 365; regression-safe imputation). Use `has_baseline_purchase = 0` to identify non-buyers. |
| `baseline_purchase_tenure_days` | Long | model | `DATEDIFF(MAX(transaction_date), MIN(transaction_date))` for rows in the window. **0 for single-purchase households** (first = last date) and **0 for non-buyers** (no dates to diff). Distinguish non-buyers from single-purchase households using `has_baseline_purchase`. |
| `baseline_12m_avg_order_value` | Double | model | `baseline_12m_revenue_sum / baseline_12m_orders` when `baseline_12m_orders > 0`; 0.0 otherwise. |
| `baseline_12m_avg_items_per_order` | Double | model | `baseline_12m_quantity_sum / baseline_12m_orders` when `baseline_12m_orders > 0`; 0.0 otherwise. |

---

### 60-Day Baseline Purchase Behavior

Sub-window: **2025-09-13 through 2025-11-11** (last 60 days of the 12-month baseline). All columns default to 0 / 0.0.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `baseline_60d_orders` | Long | model | `COUNT(DISTINCT order_id)` where `transaction_date` is in the 60-day window. |
| `baseline_60d_revenue` | Double | model | `SUM(transaction_amount)` for positive-amount rows in the 60-day window. |
| `baseline_60d_quantity` | Double | model | `SUM(quantity)` for rows in the 60-day window. |

---

### Campaign Product Focus

Filters to rows where `is_campaign_product = 1` within the 12-month baseline window. All columns default to 0 / 0.0 / `"missing"`.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `baseline_campaign_product_orders` | Long | model | `COUNT(DISTINCT order_id)` for campaign-product rows in the 12-month window. |
| `baseline_campaign_product_revenue` | Double | model | `SUM(transaction_amount)` for positive-amount campaign-product rows in the 12-month window. |
| `baseline_campaign_product_quantity` | Double | model | `SUM(quantity)` for campaign-product rows in the 12-month window. |
| `campaign_product_revenue_share` | Double | model | `baseline_campaign_product_revenue / baseline_12m_revenue` when `baseline_12m_revenue_sum > 0`; 0.0 otherwise. |
| `campaign_product_affinity_label` | String | model | `repeat_campaign_product_buyer` if `baseline_campaign_product_orders > 1`; `single_campaign_product_buyer` if = 1; `no_prior_campaign_product` otherwise. |
| `prior_campaign_product_buyer` | Integer | model | 1 if `baseline_campaign_product_orders > 0`, else 0. |

---

### Recency Ratios

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `recent_60d_revenue_share` | Double | model | `baseline_60d_revenue / baseline_12m_revenue` when `baseline_12m_revenue_sum > 0`; 0.0 otherwise. Share of annual revenue concentrated in the last 60 days. |
| `recent_60d_order_share` | Double | model | `baseline_60d_orders / baseline_12m_orders` when `baseline_12m_orders > 0`; 0.0 otherwise. |

---

### Binary Purchase Flags

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `recent_60d_buyer` | Integer | model | 1 if `baseline_60d_orders > 0`, else 0. |
| `lapsed_60d_buyer` | Integer | model | 1 if `baseline_12m_orders > 0` AND `baseline_60d_orders = 0`, else 0. Identifies households that bought in the 12-month window but not in the last 60 days. |

---

### Outcome Variables *(never used as covariates)*

Measured during the campaign period: **2025-11-12 through 2026-01-06**. Filtered to `is_campaign_product = 1`.

| Column | Type | Use | Computation |
|--------|------|-----|-------------|
| `outcome_campaign_product_orders` | Long | outcome | `COUNT(DISTINCT order_id)` for campaign-product rows in the campaign window. |
| `outcome_campaign_product_revenue` | Double | outcome | `SUM(transaction_amount)` for positive-amount campaign-product rows in the campaign window. |
| `outcome_campaign_product_quantity` | Double | outcome | `SUM(quantity)` for campaign-product rows in the campaign window. |
| `outcome_campaign_product_buyer` | Integer | outcome | 1 if `outcome_campaign_product_orders > 0`, else 0. Primary binary outcome for lift models. |

---

## Model Training Feature Summary

The following 82 columns are used as covariates in model training (all others are excluded via `MODEL_FEATURE_EXCLUDE_COLUMNS`).

| # | Column | Type | Category |
|---|--------|------|----------|
| 1 | `poc_label` | String | Stratum label |
| 2 | `state_label` | String | Stratum label |
| 3 | `baseline_buyer_label` | String | Stratum label |
| 4 | `baseline_12m_revenue_bin` | String | Stratum label |
| 5 | `num_male_in_addresslink` | Long | Gender |
| 6 | `num_female_in_addresslink` | Long | Gender |
| 7 | `num_unknown_or_other_gender_in_addresslink` | Long | Gender |
| 8 | `hh_income_code_profile_label` | String | Income |
| 9 | `num_hh_income_missing_in_addresslink` | Long | Income |
| 10 | `num_hh_income_zero_or_negative_in_addresslink` | Long | Income |
| 11–45 | `num_hh_income_code_1_in_addresslink` … `num_hh_income_code_35_in_addresslink` | Long | Income |
| 46 | `num_hh_income_code_other_in_addresslink` | Long | Income |
| 47 | `age_bucket_profile_label` | String | Age |
| 48 | `age_missing_count` | Long | Age |
| 49 | `age_lt_18_count` | Long | Age |
| 50 | `age_18_24_count` | Long | Age |
| 51 | `age_25_34_count` | Long | Age |
| 52 | `age_35_44_count` | Long | Age |
| 53 | `age_45_54_count` | Long | Age |
| 54 | `age_55_64_count` | Long | Age |
| 55 | `age_65_74_count` | Long | Age |
| 56 | `age_75_84_count` | Long | Age |
| 57 | `age_85_plus_count` | Long | Age |
| 58 | `baseline_12m_orders` | Long | 12m baseline |
| 59 | `baseline_12m_revenue_sum` | Double | 12m baseline |
| 60 | `baseline_12m_quantity_sum` | Double | 12m baseline |
| 61 | `baseline_12m_negative_transaction_count` | Long | 12m baseline |
| 62 | `baseline_12m_distinct_banners` | Long | 12m baseline |
| 63 | `baseline_12m_distinct_divisions` | Long | 12m baseline |
| 64 | `baseline_12m_active_purchase_days` | Long | 12m baseline |
| 65 | `num_weeks_purchased_in_last_365_days` | Long | 12m baseline |
| 66 | `has_baseline_purchase` | Integer | 12m baseline |
| 67 | `days_since_last_baseline_purchase` | Long | 12m baseline |
| 68 | `baseline_purchase_tenure_days` | Long | 12m baseline |
| 69 | `baseline_12m_avg_order_value` | Double | 12m baseline |
| 70 | `baseline_12m_avg_items_per_order` | Double | 12m baseline |
| 71 | `baseline_60d_orders` | Long | 60d baseline |
| 72 | `baseline_60d_revenue` | Double | 60d baseline |
| 73 | `baseline_60d_quantity` | Double | 60d baseline |
| 74 | `baseline_campaign_product_orders` | Long | Campaign product |
| 75 | `baseline_campaign_product_revenue` | Double | Campaign product |
| 76 | `baseline_campaign_product_quantity` | Double | Campaign product |
| 77 | `campaign_product_revenue_share` | Double | Campaign product |
| 78 | `campaign_product_affinity_label` | String | Campaign product |
| 79 | `prior_campaign_product_buyer` | Integer | Campaign product |
| 80 | `recent_60d_revenue_share` | Double | Recency ratio |
| 81 | `recent_60d_order_share` | Double | Recency ratio |
| 82 | `recent_60d_buyer` | Integer | Binary flag |
| 83 | `lapsed_60d_buyer` | Integer | Binary flag |

---

## Imputation & Default Values

| Column | Default | Reason |
|--------|---------|--------|
| All count columns | `0` | Zero purchases / records is the correct semantic value. |
| All revenue / quantity columns | `0.0` | Zero spend is the correct semantic value. |
| All string label columns | `"missing"` | Becomes its own category; distinguishable from real label values. |
| `days_since_last_baseline_purchase` | `366` | One day beyond the 12-month window maximum (365 days). Regression-safe: keeps non-buyers on-scale rather than as an outlier sentinel. Identify non-buyers with `has_baseline_purchase = 0`. When switching to tree models, change to `null`. |
| `baseline_purchase_tenure_days` | `0` | Undefined for non-buyers (no first/last date). `0` is shared with single-purchase households; use `has_baseline_purchase` to distinguish. |
| `has_baseline_purchase` | `0` | No purchases in the 12-month window. |

---

## Source Files

| File | Role |
|------|------|
| `FeatureEngg/custom_job/custom_code.py` | q82A — computes all features from raw inputs |
| `FeatureEnggStratify/custom_job/custom_code.py` | q84B — stratified sampling; passes q82A features through |
