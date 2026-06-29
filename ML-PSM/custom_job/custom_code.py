import pyspark.sql.functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType
from pyspark.sql.window import Window


class CustomCode:

    REBALANCE_RATIO = 2
    CALIPER = 0.01
    PS_CLIP_LOW = 0.05
    PS_CLIP_HIGH = 0.95

    def __init__(self, custom_packages_path: str = ""):
        self.custom_packages_path = custom_packages_path

    def custom_func(self, spark: SparkSession, all_features_df: DataFrame) -> DataFrame:
        """Propensity Score Matching — returns matched individual rows for ML-ATE.

        Consumes q82A AllFeatures (full pre-matched universe).
        Outputs PSMMatchedFeatures: one row per addressLink for treated +
        PSM-matched controls, with all AllFeatures columns plus propensity_score
        and treatment_group.
        """

        def optional_col(df: DataFrame, col_name: str, default_value):
            return F.col(col_name) if col_name in df.columns else default_value

        # Backward compat: pre-1.3 q82A may use old column names
        _legacy_renames = {
            "baseline_12m_revenue":     "baseline_12m_revenue_sum",
            "baseline_12m_quantity":    "baseline_12m_quantity_sum",
            "baseline_12m_revenue_bin": "baseline_12m_revenue_sum_bin",
        }
        for _old, _new in _legacy_renames.items():
            if _old in all_features_df.columns and _new not in all_features_df.columns:
                all_features_df = all_features_df.withColumnRenamed(_old, _new)

        grain_col   = "addressLink"
        partial_col = "has_partial_exposure_within_addresslink"

        # ── Step 1: Eligibility filter (same universe as q84B match_ready) ──────
        eligible = (
            all_features_df
            .filter(F.col(grain_col).isNotNull() & (F.col(grain_col) != ""))
            .filter(
                (F.col("treatment").cast(IntegerType()) == 1)
                | (
                    (F.col("treatment").cast(IntegerType()) == 0)
                    & (F.col("is_eligible_control").cast(IntegerType()) == 1)
                    & (F.coalesce(F.col(partial_col).cast(IntegerType()), F.lit(0)) == 0)
                )
            )
        )

        # ── Step 2: Ensure optional string columns exist before StringIndexer ───
        for col_name, default in [
            ("poc_label",                      "missing"),
            ("state_label",                    "missing"),
            ("baseline_buyer_label",            "missing"),
            ("campaign_product_affinity_label", "missing"),
        ]:
            if col_name not in eligible.columns:
                eligible = eligible.withColumn(col_name, F.lit(default))
            else:
                eligible = eligible.withColumn(
                    col_name,
                    F.coalesce(F.col(col_name).cast(StringType()), F.lit(default)),
                )

        # Numeric columns required by VectorAssembler — fill with 0 if absent
        numeric_feature_cols = [
            "baseline_12m_orders",
            "baseline_12m_revenue_sum",
            "baseline_12m_quantity_sum",
            "baseline_12m_negative_transaction_count",
            "baseline_12m_distinct_banners",
            "baseline_12m_distinct_divisions",
            "baseline_12m_active_purchase_days",
            "baseline_12m_avg_order_value",
            "baseline_12m_avg_items_per_order",
            "baseline_60d_orders",
            "baseline_60d_revenue",
            "baseline_60d_quantity",
            "baseline_campaign_product_orders",
            "baseline_campaign_product_revenue",
            "baseline_campaign_product_quantity",
            "has_baseline_purchase",
            "days_since_last_baseline_purchase",
            "baseline_purchase_tenure_days",
            "num_weeks_purchased_in_last_365_days",
            "recent_60d_revenue_share",
            "recent_60d_order_share",
            "campaign_product_revenue_share",
            "prior_campaign_product_buyer",
            "recent_60d_buyer",
            "lapsed_60d_buyer",
            *[f"num_hh_income_code_{i}_in_addresslink" for i in range(1, 36)],
            "num_hh_income_missing_in_addresslink",
            "num_hh_income_zero_or_negative_in_addresslink",
            "age_missing_count",
            "age_lt_18_count",
            "age_18_24_count",
            "age_25_34_count",
            "age_35_44_count",
            "age_45_54_count",
            "age_55_64_count",
            "age_65_74_count",
            "age_75_84_count",
            "age_85_plus_count",
            "num_male_in_addresslink",
            "num_female_in_addresslink",
            "num_unknown_or_other_gender_in_addresslink",
        ]
        for col_name in numeric_feature_cols:
            if col_name not in eligible.columns:
                eligible = eligible.withColumn(col_name, F.lit(0.0).cast(DoubleType()))

        categorical_index_specs = [
            ("poc_label",                      "poc_label_index"),
            ("state_label",                    "state_label_index"),
            ("baseline_buyer_label",            "baseline_buyer_label_index"),
            ("campaign_product_affinity_label", "campaign_product_affinity_label_index"),
        ]
        # handleInvalid="keep" assigns unseen labels (e.g. control-only state combos) their own index
        for input_col, output_col in categorical_index_specs:
            eligible = (
                StringIndexer(inputCol=input_col, outputCol=output_col, handleInvalid="keep")
                .fit(eligible)
                .transform(eligible)
            )

        all_feature_cols = numeric_feature_cols + [out for _, out in categorical_index_specs]

        assembler = VectorAssembler(
            inputCols=all_feature_cols,
            outputCol="features",
            handleInvalid="skip",
        )
        eligible = assembler.transform(eligible)

        # ── Step 3: Control rebalancing before LR fitting ───────────────────────
        counts = eligible.groupBy("treatment").count().collect()
        count_map = {int(row["treatment"]): row["count"] for row in counts}
        treated_count_total = int(count_map.get(1, 0))
        control_count_total = int(count_map.get(0, 0))

        treated_df = eligible.filter(F.col("treatment") == 1)
        control_df = eligible.filter(F.col("treatment") == 0)

        if (
            treated_count_total > 0
            and control_count_total > treated_count_total * self.REBALANCE_RATIO
        ):
            fraction       = min(1.0, (treated_count_total * self.REBALANCE_RATIO) / control_count_total)
            control_sample = control_df.sample(fraction=fraction, seed=42)
        else:
            control_sample = control_df

        training_df = treated_df.union(control_sample)

        # ── Step 4: Propensity score estimation ──────────────────────────────────
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="treatment",
            probabilityCol="probability",
            predictionCol="ps_prediction",
            regParam=0.1,
            elasticNetParam=0.0,   # L2 — stable under multicollinearity from 59+ correlated features
            maxIter=200,
            tol=1e-6,
        )
        lr_model = lr.fit(training_df)

        # Score the full eligible universe (not just the rebalanced training sample)
        scored = (
            lr_model.transform(eligible)
            .withColumn("propensity_score_raw", vector_to_array(F.col("probability"))[1])
            .withColumn(
                "propensity_score",
                F.when(F.col("propensity_score_raw") > self.PS_CLIP_HIGH, F.lit(self.PS_CLIP_HIGH))
                .when(F.col("propensity_score_raw") < self.PS_CLIP_LOW, F.lit(self.PS_CLIP_LOW))
                .otherwise(F.col("propensity_score_raw")),
            )
            .drop("propensity_score_raw")
        )

        # ── Step 5: Caliper bucketing matching ───────────────────────────────────
        # Integer bin avoids floating-point equality issues in groupBy/join
        scored = scored.withColumn(
            "ps_bin",
            F.floor(F.col("propensity_score") / F.lit(self.CALIPER)).cast(IntegerType()),
        )

        treated_scored = scored.filter(F.col("treatment") == 1)
        control_scored = scored.filter(F.col("treatment") == 0)

        bucket_treated_counts = (
            treated_scored
            .groupBy("ps_bin")
            .agg(F.count("*").alias("treated_in_bucket"))
        )

        # Rank controls within each bin deterministically; keep first K = treated_in_bucket
        # xxhash64 ordering matches q84B FeatureEnggStratify for consistency
        control_window = Window.partitionBy("ps_bin").orderBy(
            F.xxhash64(F.col(grain_col).cast(StringType())),
            F.col(grain_col),
        )
        matched_control = (
            control_scored
            .join(bucket_treated_counts, on="ps_bin", how="inner")
            .withColumn("control_rank", F.row_number().over(control_window))
            .filter(F.col("control_rank") <= F.col("treated_in_bucket"))
        )

        # ── Output: PSMMatchedFeatures ────────────────────────────────────────────
        # Explicit output_cols list mirrors FeatureEngg and FeatureEnggStratify style:
        # every column is named, typed, and ordered. optional_col handles columns
        # absent in older q82A versions. propensity_score and treatment_group are
        # PSM-specific additions consumed by ML-ATE.

        def opt(col_name, default_value):
            return optional_col(scored, col_name, default_value)

        output_cols = [
            F.col("addressLink").cast(StringType()),
            F.col("treatment").cast(IntegerType()),
            F.col("is_eligible_control").cast(IntegerType()),
            F.col("has_partial_exposure_within_addresslink").cast(IntegerType()),
            F.col("min_exposure_ts").cast(LongType()),
            F.col("exposure_frequency_deduped").cast(LongType()),
            F.col("mapped_online_identity_count").cast(LongType()),
            F.col("exposed_online_identity_count").cast(LongType()),
            F.col("hhpel_count").cast(LongType()),
            F.col("person_record_count").cast(LongType()),
            F.col("online_identity_count").cast(LongType()),
            F.col("poc_label").cast(StringType()),
            opt("hh_income_code_profile_label", F.lit("missing")).cast(StringType()).alias("hh_income_code_profile_label"),
            opt("num_hh_income_missing_in_addresslink", F.lit(0)).cast(LongType()).alias("num_hh_income_missing_in_addresslink"),
            opt("num_hh_income_zero_or_negative_in_addresslink", F.lit(0)).cast(LongType()).alias("num_hh_income_zero_or_negative_in_addresslink"),
            *[
                opt(f"num_hh_income_code_{i}_in_addresslink", F.lit(0)).cast(LongType()).alias(f"num_hh_income_code_{i}_in_addresslink")
                for i in range(1, 36)
            ],
            opt("num_hh_income_code_other_in_addresslink", F.lit(0)).cast(LongType()).alias("num_hh_income_code_other_in_addresslink"),
            F.col("state_label").cast(StringType()),
            opt("num_male_in_addresslink", F.lit(0)).cast(LongType()).alias("num_male_in_addresslink"),
            opt("num_female_in_addresslink", F.lit(0)).cast(LongType()).alias("num_female_in_addresslink"),
            opt("num_unknown_or_other_gender_in_addresslink", F.lit(0)).cast(LongType()).alias("num_unknown_or_other_gender_in_addresslink"),
            opt("age_bucket_profile_label", F.lit("missing")).cast(StringType()).alias("age_bucket_profile_label"),
            opt("age_missing_count", F.lit(0)).cast(LongType()).alias("age_missing_count"),
            opt("age_lt_18_count", F.lit(0)).cast(LongType()).alias("age_lt_18_count"),
            opt("age_18_24_count", F.lit(0)).cast(LongType()).alias("age_18_24_count"),
            opt("age_25_34_count", F.lit(0)).cast(LongType()).alias("age_25_34_count"),
            opt("age_35_44_count", F.lit(0)).cast(LongType()).alias("age_35_44_count"),
            opt("age_45_54_count", F.lit(0)).cast(LongType()).alias("age_45_54_count"),
            opt("age_55_64_count", F.lit(0)).cast(LongType()).alias("age_55_64_count"),
            opt("age_65_74_count", F.lit(0)).cast(LongType()).alias("age_65_74_count"),
            opt("age_75_84_count", F.lit(0)).cast(LongType()).alias("age_75_84_count"),
            opt("age_85_plus_count", F.lit(0)).cast(LongType()).alias("age_85_plus_count"),
            F.col("baseline_12m_orders").cast(LongType()),
            F.col("baseline_12m_revenue_sum").cast(DoubleType()),
            F.col("baseline_12m_quantity_sum").cast(DoubleType()),
            opt("baseline_60d_orders", F.lit(0)).cast(LongType()).alias("baseline_60d_orders"),
            opt("baseline_60d_revenue", F.lit(0.0)).cast(DoubleType()).alias("baseline_60d_revenue"),
            opt("baseline_60d_quantity", F.lit(0.0)).cast(DoubleType()).alias("baseline_60d_quantity"),
            opt("baseline_campaign_product_orders", F.lit(0)).cast(LongType()).alias("baseline_campaign_product_orders"),
            opt("baseline_campaign_product_revenue", F.lit(0.0)).cast(DoubleType()).alias("baseline_campaign_product_revenue"),
            opt("baseline_campaign_product_quantity", F.lit(0.0)).cast(DoubleType()).alias("baseline_campaign_product_quantity"),
            opt("baseline_12m_negative_transaction_count", F.lit(0)).cast(LongType()).alias("baseline_12m_negative_transaction_count"),
            opt("baseline_12m_distinct_banners", F.lit(0)).cast(LongType()).alias("baseline_12m_distinct_banners"),
            opt("baseline_12m_distinct_divisions", F.lit(0)).cast(LongType()).alias("baseline_12m_distinct_divisions"),
            opt("baseline_12m_active_purchase_days", F.lit(0)).cast(LongType()).alias("baseline_12m_active_purchase_days"),
            opt("num_weeks_purchased_in_last_365_days", F.lit(0)).cast(LongType()).alias("num_weeks_purchased_in_last_365_days"),
            opt("has_baseline_purchase", F.lit(0)).cast(IntegerType()).alias("has_baseline_purchase"),
            opt("days_since_last_baseline_purchase", F.lit(366)).cast(LongType()).alias("days_since_last_baseline_purchase"),
            opt("baseline_purchase_tenure_days", F.lit(0)).cast(LongType()).alias("baseline_purchase_tenure_days"),
            opt("baseline_12m_avg_order_value", F.lit(0.0)).cast(DoubleType()).alias("baseline_12m_avg_order_value"),
            opt("baseline_12m_avg_items_per_order", F.lit(0.0)).cast(DoubleType()).alias("baseline_12m_avg_items_per_order"),
            opt("recent_60d_revenue_share", F.lit(0.0)).cast(DoubleType()).alias("recent_60d_revenue_share"),
            opt("recent_60d_order_share", F.lit(0.0)).cast(DoubleType()).alias("recent_60d_order_share"),
            opt("campaign_product_revenue_share", F.lit(0.0)).cast(DoubleType()).alias("campaign_product_revenue_share"),
            opt("prior_campaign_product_buyer", F.lit(0)).cast(IntegerType()).alias("prior_campaign_product_buyer"),
            opt("recent_60d_buyer", F.lit(0)).cast(IntegerType()).alias("recent_60d_buyer"),
            opt("lapsed_60d_buyer", F.lit(0)).cast(IntegerType()).alias("lapsed_60d_buyer"),
            F.col("baseline_12m_revenue_sum_bin").cast(StringType()),
            opt("campaign_product_affinity_label", F.lit("missing")).cast(StringType()).alias("campaign_product_affinity_label"),
            F.col("baseline_buyer_label").cast(StringType()),
            F.col("outcome_campaign_product_orders").cast(LongType()),
            F.col("outcome_campaign_product_revenue").cast(DoubleType()),
            F.col("outcome_campaign_product_quantity").cast(DoubleType()),
            F.col("outcome_campaign_product_buyer").cast(IntegerType()),
            # PSM-specific columns — not in AllFeatures; added here for ML-ATE diagnostics
            F.col("propensity_score").cast(DoubleType()),
            F.col("treatment_group").cast(StringType()),
        ]

        treated_output  = treated_scored.withColumn("treatment_group", F.lit("treatment"))
        control_output  = matched_control.withColumn("treatment_group", F.lit("matched_control"))

        return (
            treated_output.select(*output_cols)
            .unionByName(control_output.select(*output_cols))
            .orderBy(
                F.desc("treatment"),
                F.col("treatment_group"),
                F.col("propensity_score").desc(),
                F.col("addressLink"),
            )
        )
