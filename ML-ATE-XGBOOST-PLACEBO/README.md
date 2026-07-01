# ML-ATE-XGBOOST-PLACEBO

Pre-period falsification test for the GBT-based DML pipeline (ML-ATE-XGBOOST).

Identical to ML-ATE-XGBOOST except `outcome_total_campaign_revenue` is replaced with
`baseline_60d_revenue` before `custom_func` is called. Since the campaign had not yet
launched during the 60-day pre-period, the true causal effect is zero. A significant
τ̂ ≠ 0 signals residual pre-existing spend imbalance in the PSM-matched cohort.

**Expected result:** `lift_value ≈ 0`, `lift_p_value > 0.05`, CI spanning zero.

**Note on placebo scale:** Both the primary outcome (total campaign-period spend, ~$415/hh)
and the placebo outcome (60-day baseline spend, ~$415/hh) are all-product grocery spend over
a ~56–60 day window — the scales match, making the placebo τ̂ directly interpretable as the
pre-existing confound in dollar terms.
