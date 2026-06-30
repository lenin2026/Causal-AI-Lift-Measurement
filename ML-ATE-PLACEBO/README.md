# ML-ATE-PLACEBO — Pre-Period Falsification Test

## Purpose

This module is an exact duplicate of `ML-ATE` with one change: the outcome variable is replaced by `baseline_60d_revenue` (pre-campaign revenue) instead of `outcome_campaign_product_revenue` (campaign-period revenue). Running the same Double-ML estimator on a period when the campaign was not yet active lets us test whether the causal identification assumptions hold.

---

## What is a Pre-Period Falsification Test?

A causal model makes a strong claim: any difference in outcomes between the treatment group and the matched-control group is *caused by the ad exposure*, not by pre-existing differences between the two groups.

The placebo test challenges this claim directly. The logic is:

1. The campaign launched on a fixed date (e.g. 2024-11-12). Before that date, no one was exposed to ads.
2. We already matched treatment and control households using PSM — if the match is good, the two groups should look identical in the pre-period.
3. We run the same Double-ML ATE estimator, but now we point it at pre-campaign revenue instead of campaign-period revenue.
4. If the causal model is valid, the estimated effect τ̂ on the pre-period outcome **must be statistically indistinguishable from zero** — because no treatment occurred.

A significant non-zero τ̂ in the placebo run means one of the following is wrong:
- PSM failed to balance pre-existing spend differences between treatment and control
- There is a selection effect (households that received ads were already higher-value spenders before the campaign)
- The covariate set used in PSM does not fully capture confounders

---

## What Changes vs. ML-ATE

| | ML-ATE | ML-ATE-PLACEBO |
|---|---|---|
| Outcome variable | `outcome_campaign_product_revenue` | `baseline_60d_revenue` (remapped to same column name) |
| Measurement window | Campaign period | Pre-campaign 60-day window |
| Expected τ̂ | Significant positive lift | ~0 (null effect) |
| Purpose | Estimate true causal lift | Validate PSM balance + DML identification |

The column remap happens in `client/transformation.py` before `custom_func` is called:

```python
psm_matched_features_df = psm_matched_features_df.withColumn(
    "outcome_campaign_product_revenue",
    F.col("baseline_60d_revenue").cast(DoubleType()),
)
```

This is deliberately invisible to `custom_code.py` — it reads `outcome_campaign_product_revenue` in both cases and runs the same pipeline. The swap happens upstream.

---

## How to Interpret Results

### Pass — Identification is valid

```
placebo τ̂ ≈ 0,  p-value > 0.05
```

The PSM-matched groups had similar spend levels before the campaign. Any lift measured by ML-ATE in the campaign window is plausibly causal.

### Fail — Pre-existing imbalance detected

```
placebo τ̂ significantly ≠ 0
```

The two groups were already different before the campaign started. This means:
- The PSM match may be selecting on spend propensity rather than purely on exposure probability
- Lift estimates from ML-ATE should be treated with caution and the PSM feature set should be reviewed
- Consider adding pre-period spend levels (`baseline_60d_revenue`, `baseline_90d_revenue`) as balancing covariates in FeatureEngg and re-running PSM

---

## Pipeline Position

```
FeatureEngg (q82A) ──► ML-PSM (q85A) ──► PSMMatchedFeatures
                                              │
                                              ├──► ML-ATE (q86A)          ← campaign-period lift
                                              └──► ML-ATE-PLACEBO (q86B)  ← pre-period placebo check
```

Both modules consume the same `PSMMatchedFeatures` dataset. They can run in parallel. The placebo node does not feed into any downstream computation — it is diagnostic only.

---

## Output Schema

The output schema is identical to ML-ATE (same 24 columns, same types). The only semantic difference is that `lift_value`, `lift_percent`, `avg_treatment_spend`, and `expected_amount` now reflect pre-period spend patterns. In a valid experiment, `lift_value` and `lift_percent` should be near zero and confidence intervals should contain zero.

---

## Required Input Column

`PSMMatchedFeatures` must contain `baseline_60d_revenue` (DoubleType). This column is produced by `FeatureEngg (q82A)` and passed through ML-PSM. If the column is missing, the placebo node will fail with an `AnalysisException` at the `withColumn` call in `transformation.py`.

---

## Folder Structure

```
ML-ATE-PLACEBO/
├── README.md
├── __init__.py
├── client/
│   ├── __init__.py
│   ├── data_handler.py
│   └── transformation.py   ← outcome remap lives here
├── custom_job/
│   ├── __init__.py
│   └── custom_code.py      ← identical to ML-ATE; unaware of the remap
├── requirements.txt
├── setup.py
├── tests/
└── version/
    └── __init__.py
```
