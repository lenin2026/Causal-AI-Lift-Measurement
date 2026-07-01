# XGBoost Feature Engineering Analysis

Feature selection, encoding decisions, and readiness assessment for replacing the
linear/logistic nuisance models in the Double-ML pipeline with PySpark
`GBTRegressor` + `GBTClassifier`.

---

## Current linear baseline (ML-ATE) — 11 features

| Feature | Type | Notes |
|---|---|---|
| `pre_campaign_total_order_value` | Continuous | = `baseline_12m_revenue_sum` |
| `baseline_60d_revenue` | Continuous | 60-day pre-campaign spend |
| `baseline_purchase_tenure_days` | Integer count | Customer age at retailer |
| `days_since_last_baseline_purchase` | Integer | 366 = non-buyer sentinel |
| `baseline_12m_avg_order_value` | Continuous | Average basket size |
| `baseline_12m_avg_items_per_order` | Continuous | Items per basket |
| `baseline_12m_orders` | Integer count | Transaction frequency |
| `est_age` | Continuous | Weighted midpoint across 8 age buckets |
| `est_income_code` | Continuous | Weighted average of 35 income codes (1–35) |
| `gender_index` | Label-encoded | StringIndexer on {male, female, unknown} |
| `state_index` | Label-encoded | StringIndexer on 50+ US states |

---

## Why GBT needs a different feature strategy

Linear models express the outcome as a weighted sum of features — interactions and
non-linear effects require manual feature engineering. GBT discovers both automatically
through tree splits. This changes what features to include and how to encode them.

### Aggregates vs. raw distributions

| What linear baseline uses | Problem | What GBT should use |
|---|---|---|
| `est_age` — single weighted midpoint | A household of one 25-year-old and one 65-year-old both average to ~45; the bimodal shape is lost | 8 raw age bucket counts (`age_18_24_count` … `age_85_plus_count`) |
| `est_income_code` — weighted average of 35 codes | A code-10/code-30 split household averages to 20; identical to a homogeneous code-20 household | 35 raw income code counts (`num_hh_income_code_1_in_addresslink` … `_35_`) |

GBT can split on "age_55_64_count > 1" or "num_hh_income_code_28_in_addresslink > 0" directly,
capturing household composition effects that a scalar aggregate cannot express.

---

## Encoding decisions

### ✅ Acceptable as-is (StringIndexer frequency encoding)

| Feature | Type | Reason |
|---|---|---|
| `gender` → `gender_index` | Nominal, 3 values | Frequency ordering has no wrong ordinal implication; GBT finds correct thresholds |
| `poc_label` → `poc_index` | Nominal, 3 values | Same reasoning |
| `campaign_product_affinity_label` → `affinity_index` | Nominal, 3 values | Same reasoning |

### ⚠️ Problematic with StringIndexer — manually fixed

| Feature | Problem | Fix applied in `custom_code.py` |
|---|---|---|
| `baseline_12m_revenue_sum_bin` | 13 ordinal tiers (zero … 12000_plus). StringIndexer assigns indices by frequency, destroying dollar-value order (e.g., rarest tier could be 0, most common could be 12 — reversed) | **Manual ordinal map:** zero=0, lt_10=1, …, 12000_plus=12 → `revenue_bin_ordinal` |
| `baseline_buyer_label` | 3 ordinal tiers: no_12m_purchase < lapsed_buyer < recent_buyer. Frequency-based encoding loses the recency gradient | **Manual ordinal map:** no_12m=0, lapsed=1, recent=2 → `buyer_label_ordinal` |

### ℹ️ State encoding — label index kept

`state_label` (50+ US states) is nominal — no natural ordering. StringIndexer assigns
frequency-based integers (most common state = 0). GBT treats this as a numeric split
(`state_index < 23.5`), which has no geographic meaning.

The ideal fix is one-hot encoding (50 binary columns). However, PySpark's GBT is more
efficient with label-encoded integers than OHE for high-cardinality categoricals — it
learns effective state groupings through multiple splits. **`state_index` is kept** as
StringIndexer output; GBT will form meaningful groupings regardless.

---

## New features added (not in linear baseline)

### Behavioral / conversion features

| Feature | Type | Relevance |
|---|---|---|
| `baseline_campaign_product_revenue` | Continuous | Best predictor of campaign-product outcome; prior spend on the same promoted category |
| `prior_campaign_product_buyer` | Binary (0/1) | Whether household bought campaign products in baseline year |
| `has_baseline_purchase` | Binary (0/1) | Buyer vs. non-buyer flag |
| `baseline_12m_quantity_sum` | Integer | Volume signal independent of price |
| `baseline_60d_orders` | Integer | Recent transaction frequency |
| `recent_60d_buyer` | Binary (0/1) | Active in last 60 days |
| `lapsed_60d_buyer` | Binary (0/1) | Bought in 12m but not in last 60 days |
| `campaign_product_revenue_share` | Ratio | Fraction of basket that is campaign-product category |
| `recent_60d_revenue_share` | Ratio | 60d spend / 12m spend — recency trend signal |
| `recent_60d_order_share` | Ratio | 60d orders / 12m orders |

### Raw age bucket counts (8 features)

Replace scalar `est_age`. GBT can discover segment-specific effects:
`age_18_24_count`, `age_25_34_count`, `age_35_44_count`, `age_45_54_count`,
`age_55_64_count`, `age_65_74_count`, `age_75_84_count`, `age_85_plus_count`

### Raw income code counts (35 features)

Replace scalar `est_income_code`. LR IXI income codes 1–35 (higher = higher income).
GBT exploits distributional shape (e.g., presence of multiple high-code records signals
high-income multi-person household):
`num_hh_income_code_1_in_addresslink` … `num_hh_income_code_35_in_addresslink`

### Exposure features (2 features)

| Feature | Relevance |
|---|---|
| `exposure_frequency_deduped` | Ad dose — could be a confounder if heavy exposure correlates with spend |
| `person_record_count` | Household size proxy |

---

## Missing value handling

FeatureEngg fills all nulls before data reaches ML-ATE-XGBOOST:
- Integer counts → 0
- Revenue/ratio doubles → 0.0
- String labels → "missing" (handled by StringIndexer `handleInvalid="keep"`)
- Non-buyers: `days_since_last_baseline_purchase` = 366

VectorAssembler uses `handleInvalid="keep"` (not "skip") so no rows are dropped.
GBT handles NaN natively by learning optimal split direction for missing values.

---

## Feature scaling

Tree-based models are **scale-invariant** — no normalization or standardization is
applied or needed. Raw dollar values ($0–$50,000+), counts (0–500), and ratios (0–1)
can coexist in the same assembler without normalization. Applying StandardScaler would
be wasted computation and would not change GBT results.

---

## Package dependency

No additional packages are required. `GBTRegressor` and `GBTClassifier` are part of
`pyspark.ml` which is always available in Habu Clean Compute. External `xgboost`
package is **not** used.

If future versions need true XGBoost (faster training, native categorical support),
add `xgboost>=2.0` to `requirements.txt` and replace:
- `GBTRegressor` → `xgboost.spark.SparkXGBRegressor`
- `GBTClassifier` → `xgboost.spark.SparkXGBClassifier`

---

## Feature count summary

| Group | Count |
|---|---|
| Spend / frequency (continuous) | 13 |
| Binary engagement flags | 4 |
| Raw age bucket counts | 8 |
| Raw income code counts | 35 |
| Ordinal-encoded categoricals | 2 |
| Frequency-indexed nominal categoricals | 4 |
| Exposure signals | 2 |
| **Total** | **68** |

Compared to the linear baseline's 11 features: **+57 features**, all drawn from
columns already present in `PSMMatchedFeatures` / `StratifiedFeatures` — no
FeatureEngg changes required.
