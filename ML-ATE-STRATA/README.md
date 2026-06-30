# ML-ATE — Double-ML Average Treatment Effect Estimator

Estimates the causal lift of an advertising campaign using Double Machine Learning (DML)
with K=5 cross-fitting. Input is `PSMMatchedFeatures` (output of ML-PSM). Output is a
single summary row with ATE, CATE by segment, confidence intervals, and model diagnostics.

---

## Notation Reference

| Symbol | Meaning | In this pipeline |
|---|---|---|
| `T` | Treatment indicator | `treatment` column: 1 = ad-exposed household, 0 = matched control |
| `T=1` | Treated unit | Household that received at least one ad impression (set by q80A) |
| `T=0` | Control unit | Household with no ad exposure; eligible control candidate |
| `Y` | Outcome variable | `post_campaign_total_order_value` — campaign-period spend |
| `Y(1)` | Potential outcome under treatment | Spend the household *would have* if exposed |
| `Y(0)` | Potential outcome under control | Spend the household *would have* if not exposed |
| `X` | Covariate vector | Baseline features fed to VectorAssembler (spend history, demographics, etc.) |
| `τ` (tau) | Average Treatment Effect (ATE) | `incremental_lift` — average causal effect of ad exposure on spend |
| `τ̂` (tau-hat) | Estimated ATE | The value ML-ATE computes; an estimate of the true τ |
| `E[Y]` | Expected value (population average) of Y | Average spend across all units |
| `E[Y\|T=1]` | Conditional expectation of Y given T=1 | Average spend among exposed households (`avg_treatment_amount`) |
| `E[Y(1)]` | Expected potential outcome under treatment | Average spend if everyone were exposed |
| `E[Y(0)]` | Expected potential outcome under control | Average spend if no one were exposed |
| `Ŷ` (Y-hat) | Predicted outcome from outcome model | Output of `lr_y` (Huber regression) in cross-fitting |
| `T̂` (T-hat) | Predicted treatment propensity | Output of `lr_t` (Logistic Regression) in cross-fitting |
| `ỹ` / `y_res` | Outcome residual | `Y − Ŷ` — variation in spend not explained by covariates |
| `t̃` / `t_res` | Treatment residual | `T − T̂` — variation in treatment not explained by covariates |
| `K` | Number of cross-fitting folds | 5 |
| `p-value` | Probability of observing τ̂ if true τ = 0 | `lift_p_value`; below 0.05 = statistically significant lift |
| `CI` | Confidence interval | `[lift_ci_lower, lift_ci_upper]` at 95% (±1.96 × standard error) |
| `CATE` | Conditional Average Treatment Effect | Lift estimate within a specific segment (e.g. buyers, young, high income) |

### Key identities used in the output

```
τ = E[Y(1)] − E[Y(0)]                          # ATE definition (incremental_lift)

expected_amount = E[Y|T=1] − τ                  # counterfactual baseline spend
                                                 # ≈ what the treatment group would
                                                 # have spent without the ads

lift_percent = τ / expected_amount × 100        # % lift over counterfactual baseline

CATE_segment = τ_base + δ_segment               # segment lift = base ATE + interaction term
```

### Why unconditional E[Y|T=1] (not E[Y|T=1, Y>0])

`avg_treatment_amount` uses the full exposed population including $0 spenders.
Using only converters (Y>0) would compute E[Y|T=1, Y>0], which is 7–20× larger
for a typical 5–15% conversion rate campaign. Subtracting an unconditional τ from
a conditional average breaks the estimand match and deflates `lift_percent`.

---

## Pipeline Position

```
PSMMatchedFeatures (from ML-PSM)
    └─► ML-ATE ──► single-row lift summary
```

**Input macro:** `PSMMatchedFeatures`
**Output:** 24-column summary row (see output schema in root README)

---

## Model Pipeline

1. **Column mapping** — `outcome_campaign_product_revenue` → `post_campaign_total_order_value`; `baseline_12m_revenue_sum` → `pre_campaign_total_order_value`
2. **Derived demographics** — `est_age` (weighted midpoint from age-bucket counts); `est_income_code` (weighted average of 35 income codes); majority `gender`
3. **99th-percentile winsorisation** — caps extreme spend outliers
4. **K=5 cross-fitting** — outcome model (`lr_y`, Huber regression) and treatment model (`lr_t`, Logistic Regression) fitted on 4 folds, predictions on held-out fold; repeated for all 5 folds
5. **Residualisation** — `y_res = Y − Ŷ`; `t_res = T − T̂`
6. **ATE** — OLS of `y_res ~ t_res` → coefficient is τ̂
7. **CATE** — interaction terms `t_res × segment` for 5 segments: existing buyer, lapsed, young, senior, high income

---

## Folder Structure

```
ML-ATE/
├── README.md
├── __init__.py
├── client/
│   ├── __init__.py
│   ├── data_handler.py
│   └── transformation.py
├── custom_job/
│   ├── __init__.py
│   └── custom_code.py
├── requirements.txt
├── setup.py
├── tests/
└── version/
    └── __init__.py
```
