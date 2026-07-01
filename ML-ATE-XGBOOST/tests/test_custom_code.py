"""
Unit tests for ML-ATE-XGBOOST custom_code.py.

Run from repo root:
    python3 -m pytest ML-ATE-XGBOOST/tests/test_custom_code.py -v

Or directly:
    cd ML-ATE-XGBOOST && python3 tests/test_custom_code.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DoubleType, IntegerType, LongType, StringType,
)

from custom_job.custom_code import CustomCode

# ---------------------------------------------------------------------------
# Spark session (local, single-threaded for tests)
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("test_xgboost_ate") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


# ---------------------------------------------------------------------------
# Synthetic DataFrame factory
# ---------------------------------------------------------------------------

def _make_df(n=120, use_new_outcome_col=True, placebo=False):
    """
    Build a minimal synthetic PSMMatchedFeatures-like DataFrame with all columns
    that custom_func reads. n must be large enough for K=5 cross-fitting (>=10).
    """
    random.seed(42)

    # income code counts — 35 cols, sparse
    income_cols = {f"num_hh_income_code_{i}_in_addresslink": 0 for i in range(1, 36)}

    rows = []
    for i in range(n):
        treat = i % 2  # alternating treatment/control
        spend = round(random.uniform(0, 500), 2)
        b60d  = round(random.uniform(0, 100), 2)
        outcome_total = round(random.uniform(0, 600), 2) if treat else round(random.uniform(0, 400), 2)

        income_row = dict(income_cols)
        income_row[f"num_hh_income_code_{random.randint(1, 35)}_in_addresslink"] = 1

        row = {
            # outcome columns
            "outcome_total_campaign_revenue":   outcome_total if (use_new_outcome_col and not placebo) else None,
            "outcome_campaign_product_revenue": round(random.uniform(0, 10), 2),
            "outcome_campaign_product_orders":  random.randint(0, 3),
            "outcome_campaign_product_buyer":   random.randint(0, 1),
            # baseline spend
            "baseline_12m_revenue_sum":    spend,
            "baseline_60d_revenue":        b60d,
            "baseline_12m_orders":         random.randint(0, 20),
            "baseline_12m_quantity_sum":   float(random.randint(0, 50)),
            "baseline_60d_orders":         random.randint(0, 5),
            "baseline_campaign_product_revenue": round(random.uniform(0, 20), 2),
            "baseline_12m_avg_order_value":     round(spend / max(random.randint(1, 10), 1), 2),
            "baseline_12m_avg_items_per_order": round(random.uniform(1, 5), 2),
            "baseline_purchase_tenure_days":    random.randint(0, 365),
            "days_since_last_baseline_purchase": random.randint(0, 366),
            # revenue bin
            "baseline_12m_revenue_sum_bin": random.choice([
                "zero", "lt_10", "10_to_49", "50_to_99", "100_to_249",
                "250_to_499", "500_to_999", "1000_to_1999",
            ]),
            # buyer label
            "baseline_buyer_label": random.choice(["recent_buyer", "lapsed_buyer", "no_12m_purchase"]),
            # flags
            "has_baseline_purchase":       random.randint(0, 1),
            "prior_campaign_product_buyer": random.randint(0, 1),
            "recent_60d_buyer":            random.randint(0, 1),
            "lapsed_60d_buyer":            random.randint(0, 1),
            # engagement ratios
            "campaign_product_revenue_share": round(random.uniform(0, 0.3), 4),
            "recent_60d_revenue_share":       round(random.uniform(0, 1), 4),
            "recent_60d_order_share":         round(random.uniform(0, 1), 4),
            # age buckets
            "age_18_24_count": random.randint(0, 2),
            "age_25_34_count": random.randint(0, 2),
            "age_35_44_count": random.randint(0, 2),
            "age_45_54_count": random.randint(0, 2),
            "age_55_64_count": random.randint(0, 2),
            "age_65_74_count": random.randint(0, 2),
            "age_75_84_count": random.randint(0, 2),
            "age_85_plus_count": random.randint(0, 1),
            # demographics
            "num_male_in_addresslink":   random.randint(0, 3),
            "num_female_in_addresslink": random.randint(0, 3),
            "state_label":    random.choice(["CA", "TX", "NY", "FL", "WA"]),
            "poc_label":      random.choice(["has_child", "no_child", "missing"]),
            "campaign_product_affinity_label": random.choice([
                "repeat_campaign_product_buyer",
                "single_campaign_product_buyer",
                "no_prior_campaign_product",
            ]),
            # exposure
            "exposure_frequency_deduped": float(random.randint(0, 10)),
            "person_record_count":        float(random.randint(1, 4)),
            # treatment
            "treatment": treat,
        }
        row.update(income_row)
        rows.append(row)

    # Drop outcome_total_campaign_revenue if testing the fallback path
    if not use_new_outcome_col:
        for r in rows:
            del r["outcome_total_campaign_revenue"]

    return spark.createDataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

EXPECTED_OUTPUT_COLS = {
    "incremental_lift", "lift_percent", "avg_treatment_amount", "expected_amount",
    "total_row_count", "treated_count", "outcome_r2", "treatment_auc",
    "lift_pct_ci_lower", "lift_pct_ci_upper", "lift_ci_lower", "lift_ci_upper",
    "lift_p_value", "ate_base",
    "cate_existing_buyer", "cate_lapsed", "cate_young", "cate_senior", "cate_high_income",
    "cate_existing_buyer_p_value", "cate_lapsed_p_value", "cate_young_p_value",
    "cate_senior_p_value", "cate_high_income_p_value",
}


def test_output_schema_with_new_outcome_col():
    """Primary path: outcome_total_campaign_revenue present."""
    df = _make_df(use_new_outcome_col=True)
    result = CustomCode().custom_func(spark, df)
    assert set(result.columns) == EXPECTED_OUTPUT_COLS, (
        f"Schema mismatch. Got: {set(result.columns)}"
    )
    assert result.count() == 1
    print("PASS test_output_schema_with_new_outcome_col")


def test_output_schema_fallback_to_campaign_product_revenue():
    """Fallback path: outcome_total_campaign_revenue absent — must use outcome_campaign_product_revenue."""
    df = _make_df(use_new_outcome_col=False)
    assert "outcome_total_campaign_revenue" not in df.columns
    result = CustomCode().custom_func(spark, df)
    assert set(result.columns) == EXPECTED_OUTPUT_COLS, (
        f"Schema mismatch on fallback path. Got: {set(result.columns)}"
    )
    assert result.count() == 1
    print("PASS test_output_schema_fallback_to_campaign_product_revenue")


def test_single_output_row():
    """Output must be exactly one row (summary statistics)."""
    df = _make_df()
    result = CustomCode().custom_func(spark, df)
    assert result.count() == 1
    print("PASS test_single_output_row")


def test_output_values_are_finite():
    """All 24 output columns must be non-null finite floats."""
    df = _make_df()
    result = CustomCode().custom_func(spark, df)
    row = result.collect()[0]
    for col in EXPECTED_OUTPUT_COLS:
        val = row[col]
        assert val is not None, f"Column {col} is None"
        assert val == val, f"Column {col} is NaN"  # NaN != NaN
    print("PASS test_output_values_are_finite")


def test_lift_percent_sign():
    """lift_percent = incremental_lift / expected_amount * 100 — signs must match."""
    df = _make_df()
    result = CustomCode().custom_func(spark, df)
    row = result.collect()[0]
    if row["expected_amount"] != 0:
        expected_sign = row["incremental_lift"] / row["expected_amount"]
        actual_sign   = row["lift_percent"] / 100
        assert (expected_sign >= 0) == (actual_sign >= 0), (
            f"lift_percent sign inconsistency: lift={row['incremental_lift']}, "
            f"expected={row['expected_amount']}, pct={row['lift_percent']}"
        )
    print("PASS test_lift_percent_sign")


if __name__ == "__main__":
    tests = [
        test_output_schema_with_new_outcome_col,
        test_output_schema_fallback_to_campaign_product_revenue,
        test_single_output_row,
        test_output_values_are_finite,
        test_lift_percent_sign,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    spark.stop()
    if failed:
        print(f"\n{len(failed)} test(s) FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed.")
