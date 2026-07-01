# ML-ATE-XGBOOST

Double-ML ATE estimation using **PySpark GBTRegressor + GBTClassifier** as nuisance models
in place of the linear/logistic baseline used by ML-ATE.

Same input (`PSMMatchedFeatures`), same output schema (24 columns), same K=5 cross-fitting
structure — only the nuisance model family changes.

See `XGBoostFeatureEngineering.md` for the full feature-selection and encoding rationale.

## Why GBT over linear nuisance models

| Dimension | Linear baseline (ML-ATE) | GBT (this node) |
|---|---|---|
| Outcome model | Huber regression (11 features) | GBTRegressor, absolute loss (68 features) |
| Treatment model | Logistic regression (11 features) | GBTClassifier (68 features) |
| Feature interactions | None — linear only | Learned automatically up to `maxDepth=5` |
| Age / income | Weighted aggregate scalars | Raw bucket/code counts — preserves distribution shape |
| Ordinal categoricals | Not included | Manually ordinal-encoded (revenue bin, buyer label) |
| Missing values | Rows dropped (handleInvalid=skip) | Handled natively; no rows dropped |
| Outcome R² expected | Near zero (sparse outcome kills linear) | Substantially higher — non-linear splits handle zero-inflation |

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `maxIter` | 50 | Number of boosting trees; balances accuracy and Habu job time |
| `maxDepth` | 5 | Captures up to 5-way feature interactions |
| `stepSize` | 0.1 | Standard learning rate; conservative to avoid overfitting |
| `subsamplingRate` | 0.8 | Row subsampling per tree; reduces variance |
| `featureSubsetStrategy` | auto | sqrt(n) for classifier, n/3 for regressor |
| `lossType` (regressor) | absolute | L1 loss — robust to outlier households; mirrors Huber |
| `seed` | 42 | Reproducibility |

## Feature set (68 features)

| Group | Count | Columns |
|---|---|---|
| Spend / frequency | 13 | pre_campaign_total_order_value, baseline_60d_revenue, tenure, recency, avg_order_value, avg_items, orders, quantity, 60d_orders, campaign_product_revenue, revenue shares |
| Binary flags | 4 | has_baseline_purchase, prior_campaign_product_buyer, recent_60d_buyer, lapsed_60d_buyer |
| Raw age buckets | 8 | age_18_24_count … age_85_plus_count |
| Raw income codes | 35 | num_hh_income_code_1 … _35_in_addresslink |
| Ordinal categoricals | 2 | revenue_bin_ordinal (0–12), buyer_label_ordinal (0–2) |
| Nominal indexed | 4 | gender_index, state_index, poc_index, affinity_index |
| Exposure | 2 | exposure_frequency_deduped, person_record_count |

## Output schema

Identical 24-column schema as ML-ATE — results are directly comparable row-for-row.
