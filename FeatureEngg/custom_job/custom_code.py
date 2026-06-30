import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType


class CustomCode:
    def __init__(self, custom_packages_path: str = ""):
        # The root directory where the custom artifact is installed.
        self.custom_packages_path = custom_packages_path

    def custom_func(
        self,
        spark: SparkSession,
        conversion_df: DataFrame,
        exposure_df: DataFrame,
        demographic_df: DataFrame,
    ) -> DataFrame:
        grain_col = "addressLink"

        def require_columns(df: DataFrame, required_columns, df_name: str) -> None:
            missing_columns = [col_name for col_name in required_columns if col_name not in df.columns]
            if missing_columns:
                available_columns = ", ".join(df.columns[:40])
                raise ValueError(
                    f"{df_name} missing required columns: {', '.join(missing_columns)}. "
                    f"For q82A, conversion_df must be the q79A conversion_rows_mapped_to_addresslink output, "
                    f"not raw conversion rows. Available columns: {available_columns}"
                )

        def canonicalize_columns(df: DataFrame, canonical_columns) -> DataFrame:
            actual_by_lower = {col_name.lower(): col_name for col_name in df.columns}
            for canonical_col in canonical_columns:
                actual_col = actual_by_lower.get(canonical_col.lower())
                if actual_col and actual_col != canonical_col and canonical_col not in df.columns:
                    df = df.withColumnRenamed(actual_col, canonical_col)
            return df

        def optional_col(df: DataFrame, col_name: str, default_value):
            return F.col(col_name) if col_name in df.columns else default_value

        def optional_flag_col(df: DataFrame, col_name: str, default_value: int = 0):
            if col_name not in df.columns:
                return F.lit(default_value).cast(IntegerType())
            value_text = F.lower(F.trim(F.col(col_name).cast(StringType())))
            return (
                F.when(value_text.isin("1", "true", "t", "yes", "y"), F.lit(1))
                .when(value_text.isin("0", "false", "f", "no", "n"), F.lit(0))
                .otherwise(F.coalesce(F.col(col_name).cast(IntegerType()), F.lit(default_value)))
                .cast(IntegerType())
            )

        def count_distinct_optional(df: DataFrame, col_name: str, alias_name: str, condition=None):
            if col_name not in df.columns:
                return F.max(F.lit(0)).cast(LongType()).alias(alias_name)
            value = F.col(col_name)
            if condition is not None:
                value = F.when(condition, value)
            return F.countDistinct(value).cast(LongType()).alias(alias_name)

        conversion_df = canonicalize_columns(
            conversion_df,
            [
                grain_col,
                "order_id",
                "transaction_amount",
                "quantity",
                "is_campaign_product",
                "transaction_date",
                "transaction_timestamp_unix",
                "banner",
                "division",
                "product_id",
                "product_brand",
                "transaction_category",
            ],
        )
        exposure_df = canonicalize_columns(
            exposure_df,
            [
                grain_col,
                "treatment",
                "is_eligible_control",
                "has_partial_exposure_within_addresslink",
                "min_exposure_ts",
                "exposure_frequency_deduped",
                "mapped_online_identity_count",
                "exposed_online_identity_count",
                "person_record_count",
                "hhpel_count",
            ],
        )
        demographic_df = canonicalize_columns(
            demographic_df,
            [
                grain_col,
                "rampid",
                "install_date",
                "gender",
                "age",
                "state",
                "hh_income",
                "poc",
            ],
        )

        require_columns(
            conversion_df,
            [grain_col, "order_id", "transaction_amount", "quantity", "is_campaign_product"],
            "conversion_df",
        )
        require_columns(exposure_df, [grain_col, "treatment", "is_eligible_control"], "exposure_df")
        require_columns(demographic_df, [grain_col], "demographic_df")

        if "transaction_date" not in conversion_df.columns and "transaction_timestamp_unix" not in conversion_df.columns:
            raise ValueError("conversion_df must contain transaction_date or transaction_timestamp_unix")

        # q82 extension from the existing artifact: latest demographics are kept per online identity.
        demographic_df = demographic_df.filter(
            F.col(grain_col).isNotNull() & (F.trim(F.col(grain_col).cast("string")) != "")
        )
        if "rampid" in demographic_df.columns and "install_date" in demographic_df.columns:
            demographic_df = (
                demographic_df.withColumn(
                    "rn",
                    F.row_number().over(
                        Window.partitionBy("rampid").orderBy(F.desc_nulls_last(F.col("install_date")))
                    ),
                )
                .filter(F.col("rn") == 1)
                .drop("rn")
            )
        elif "install_date" in demographic_df.columns:
            demographic_df = (
                demographic_df.withColumn(
                    "rn",
                    F.row_number().over(
                        Window.partitionBy(grain_col).orderBy(F.desc_nulls_last(F.col("install_date")))
                    ),
                )
                .filter(F.col("rn") == 1)
                .drop("rn")
            )

        for col_name in ["gender", "age", "state", "hh_income", "poc"]:
            if col_name not in demographic_df.columns:
                demographic_df = demographic_df.withColumn(col_name, F.lit(None).cast(StringType()))

        poc_text = F.lower(F.trim(F.col("poc").cast("string")))
        income_value = F.col("hh_income").cast(IntegerType())
        state_value = F.when(
            F.col("state").isNull() | (F.trim(F.col("state").cast("string")) == ""),
            F.lit(None).cast(StringType()),
        ).otherwise(F.trim(F.col("state").cast("string")))
        gender_text = F.lower(F.trim(F.col("gender").cast("string")))
        age_value = F.col("age").cast(DoubleType())

        income_count_columns = [
            "num_hh_income_missing_in_addresslink",
            "num_hh_income_zero_or_negative_in_addresslink",
            *[f"num_hh_income_code_{code}_in_addresslink" for code in range(1, 36)],
            "num_hh_income_code_other_in_addresslink",
        ]

        demographic_df = (
            demographic_df.groupBy(grain_col)
            .agg(
                F.sum(F.when(poc_text.isin("true", "1", "yes", "y"), 1).otherwise(0))
                .cast(LongType())
                .alias("has_child_count"),
                F.sum(F.when(poc_text.isin("false", "0", "no", "n"), 1).otherwise(0))
                .cast(LongType())
                .alias("no_child_count"),
                F.sum(F.when(income_value.isNull(), 1).otherwise(0))
                .cast(LongType())
                .alias("num_hh_income_missing_in_addresslink"),
                F.sum(F.when(income_value <= 0, 1).otherwise(0))
                .cast(LongType())
                .alias("num_hh_income_zero_or_negative_in_addresslink"),
                *[
                    F.sum(F.when(income_value == code, 1).otherwise(0))
                    .cast(LongType())
                    .alias(f"num_hh_income_code_{code}_in_addresslink")
                    for code in range(1, 36)
                ],
                F.sum(F.when(income_value > 35, 1).otherwise(0))
                .cast(LongType())
                .alias("num_hh_income_code_other_in_addresslink"),
                F.countDistinct(state_value).cast(LongType()).alias("state_distinct_count"),
                F.max(state_value).alias("state_max_value"),
                F.sum(F.when(gender_text.isin("m", "male"), 1).otherwise(0))
                .cast(LongType())
                .alias("num_male_in_addresslink"),
                F.sum(F.when(gender_text.isin("f", "female"), 1).otherwise(0))
                .cast(LongType())
                .alias("num_female_in_addresslink"),
                F.sum(
                    F.when(
                        gender_text.isNull() | (~gender_text.isin("m", "male", "f", "female")),
                        1,
                    ).otherwise(0)
                )
                .cast(LongType())
                .alias("num_unknown_or_other_gender_in_addresslink"),
                F.count(age_value).cast(LongType()).alias("age_known_count"),
                F.sum(F.when(age_value.isNull(), 1).otherwise(0)).cast(LongType()).alias("age_missing_count"),
                F.sum(F.when(age_value < 18, 1).otherwise(0)).cast(LongType()).alias("age_lt_18_count"),
                F.sum(F.when(age_value.between(18, 24), 1).otherwise(0))
                .cast(LongType())
                .alias("age_18_24_count"),
                F.sum(F.when(age_value.between(25, 34), 1).otherwise(0))
                .cast(LongType())
                .alias("age_25_34_count"),
                F.sum(F.when(age_value.between(35, 44), 1).otherwise(0))
                .cast(LongType())
                .alias("age_35_44_count"),
                F.sum(F.when(age_value.between(45, 54), 1).otherwise(0))
                .cast(LongType())
                .alias("age_45_54_count"),
                F.sum(F.when(age_value.between(55, 64), 1).otherwise(0))
                .cast(LongType())
                .alias("age_55_64_count"),
                F.sum(F.when(age_value.between(65, 74), 1).otherwise(0))
                .cast(LongType())
                .alias("age_65_74_count"),
                F.sum(F.when(age_value.between(75, 84), 1).otherwise(0))
                .cast(LongType())
                .alias("age_75_84_count"),
                F.sum(F.when(age_value >= 85, 1).otherwise(0)).cast(LongType()).alias("age_85_plus_count"),
            )
            .withColumn(
                "poc_label",
                F.when(F.col("has_child_count") > 0, F.lit("has_child"))
                .when(F.col("no_child_count") > 0, F.lit("no_child"))
                .otherwise(F.lit("missing")),
            )
            .withColumn(
                "state_label",
                F.when(F.col("state_distinct_count") == 0, F.lit("missing"))
                .when(F.col("state_distinct_count") > 1, F.lit("mixed_or_multiple"))
                .otherwise(F.col("state_max_value")),
            )
            .withColumn(
                "age_bucket_profile_label",
                F.when(F.col("age_known_count") == 0, F.lit("missing")).otherwise(
                    F.concat_ws(
                        "|",
                        F.when(F.col("age_lt_18_count") > 0, F.lit("lt_18")),
                        F.when(F.col("age_18_24_count") > 0, F.lit("18_24")),
                        F.when(F.col("age_25_34_count") > 0, F.lit("25_34")),
                        F.when(F.col("age_35_44_count") > 0, F.lit("35_44")),
                        F.when(F.col("age_45_54_count") > 0, F.lit("45_54")),
                        F.when(F.col("age_55_64_count") > 0, F.lit("55_64")),
                        F.when(F.col("age_65_74_count") > 0, F.lit("65_74")),
                        F.when(F.col("age_75_84_count") > 0, F.lit("75_84")),
                        F.when(F.col("age_85_plus_count") > 0, F.lit("85_plus")),
                    )
                ),
            )
        )

        income_profile_parts = [
            F.when(
                F.col("num_hh_income_missing_in_addresslink") > 0,
                F.concat(F.lit("missing:"), F.col("num_hh_income_missing_in_addresslink").cast(StringType())),
            ),
            F.when(
                F.col("num_hh_income_zero_or_negative_in_addresslink") > 0,
                F.concat(
                    F.lit("zero_or_negative:"),
                    F.col("num_hh_income_zero_or_negative_in_addresslink").cast(StringType()),
                ),
            ),
            *[
                F.when(
                    F.col(f"num_hh_income_code_{code}_in_addresslink") > 0,
                    F.concat(
                        F.lit(f"code_{code}:"),
                        F.col(f"num_hh_income_code_{code}_in_addresslink").cast(StringType()),
                    ),
                )
                for code in range(1, 36)
            ],
            F.when(
                F.col("num_hh_income_code_other_in_addresslink") > 0,
                F.concat(F.lit("other:"), F.col("num_hh_income_code_other_in_addresslink").cast(StringType())),
            ),
        ]
        income_total_count = F.lit(0)
        for col_name in income_count_columns:
            income_total_count = income_total_count + F.col(col_name)

        demographic_df = demographic_df.withColumn(
            "hh_income_code_profile_label",
            F.when(income_total_count == 0, F.lit("missing")).otherwise(F.concat_ws("|", *income_profile_parts)),
        ).select(
            grain_col,
            "poc_label",
            "hh_income_code_profile_label",
            *income_count_columns,
            "state_label",
            "num_male_in_addresslink",
            "num_female_in_addresslink",
            "num_unknown_or_other_gender_in_addresslink",
            "age_bucket_profile_label",
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
        )

        if "transaction_date" not in conversion_df.columns:
            conversion_df = conversion_df.withColumn(
                "transaction_date",
                F.to_date(F.from_unixtime(F.col("transaction_timestamp_unix").cast(LongType()))),
            )
        else:
            conversion_df = conversion_df.withColumn("transaction_date", F.to_date(F.col("transaction_date")))

        conversion_df = (
            conversion_df.withColumn("transaction_amount", F.col("transaction_amount").cast(DoubleType()))
            .withColumn("quantity", F.col("quantity").cast(DoubleType()))
            .withColumn("is_campaign_product", F.coalesce(F.col("is_campaign_product").cast(IntegerType()), F.lit(0)))
        )

        baseline_12m_w = F.col("transaction_date").between(
            F.to_date(F.lit("2024-11-12")), F.to_date(F.lit("2025-11-11"))
        )
        baseline_60d_w = F.col("transaction_date").between(
            F.to_date(F.lit("2025-09-13")), F.to_date(F.lit("2025-11-11"))
        )
        outcome_campaign_w = F.col("transaction_date").between(
            F.to_date(F.lit("2025-11-12")), F.to_date(F.lit("2026-01-06"))
        )
        campaign_product_w = F.col("is_campaign_product") == 1
        positive_amount_w = F.col("transaction_amount") > 0

        features_df = (
            conversion_df.groupBy(grain_col)
            .agg(
                F.countDistinct(F.when(baseline_12m_w, F.col("order_id")))
                .cast(LongType())
                .alias("baseline_12m_orders"),
                F.sum(
                    F.when(baseline_12m_w & positive_amount_w, F.col("transaction_amount")).otherwise(F.lit(0.0))
                ).alias("baseline_12m_revenue_sum"),
                F.sum(F.when(baseline_12m_w, F.col("quantity")).otherwise(F.lit(0.0))).alias(
                    "baseline_12m_quantity_sum"
                ),
                F.countDistinct(F.when(baseline_60d_w, F.col("order_id")))
                .cast(LongType())
                .alias("baseline_60d_orders"),
                F.sum(
                    F.when(baseline_60d_w & positive_amount_w, F.col("transaction_amount")).otherwise(F.lit(0.0))
                ).alias("baseline_60d_revenue"),
                F.sum(F.when(baseline_60d_w, F.col("quantity")).otherwise(F.lit(0.0))).alias(
                    "baseline_60d_quantity"
                ),
                F.countDistinct(F.when(baseline_12m_w & campaign_product_w, F.col("order_id")))
                .cast(LongType())
                .alias("baseline_campaign_product_orders"),
                F.sum(
                    F.when(
                        baseline_12m_w & campaign_product_w & positive_amount_w,
                        F.col("transaction_amount"),
                    ).otherwise(F.lit(0.0))
                ).alias("baseline_campaign_product_revenue"),
                F.sum(F.when(baseline_12m_w & campaign_product_w, F.col("quantity")).otherwise(F.lit(0.0))).alias(
                    "baseline_campaign_product_quantity"
                ),
                F.sum(F.when(baseline_12m_w & (F.col("transaction_amount") < 0), 1).otherwise(0))
                .cast(LongType())
                .alias("baseline_12m_negative_transaction_count"),
                count_distinct_optional(
                    conversion_df, "banner", "baseline_12m_distinct_banners", condition=baseline_12m_w
                ),
                count_distinct_optional(
                    conversion_df, "division", "baseline_12m_distinct_divisions", condition=baseline_12m_w
                ),
                F.countDistinct(F.when(baseline_12m_w, F.col("transaction_date")))
                .cast(LongType())
                .alias("baseline_12m_active_purchase_days"),
                F.countDistinct(F.when(baseline_12m_w, F.date_trunc("week", F.col("transaction_date").cast("timestamp"))))
                .cast(LongType())
                .alias("num_weeks_purchased_in_last_365_days"),
                F.min(F.when(baseline_12m_w, F.col("transaction_date"))).alias("first_baseline_purchase_date"),
                F.max(F.when(baseline_12m_w, F.col("transaction_date"))).alias("last_baseline_purchase_date"),
                F.countDistinct(F.when(outcome_campaign_w & campaign_product_w, F.col("order_id")))
                .cast(LongType())
                .alias("outcome_campaign_product_orders"),
                F.sum(
                    F.when(
                        outcome_campaign_w & campaign_product_w & positive_amount_w,
                        F.col("transaction_amount"),
                    ).otherwise(F.lit(0.0))
                ).alias("outcome_campaign_product_revenue"),
                F.sum(F.when(outcome_campaign_w & campaign_product_w, F.col("quantity")).otherwise(F.lit(0.0))).alias(
                    "outcome_campaign_product_quantity"
                ),
            )
            .withColumn(
                "days_since_last_baseline_purchase",
                F.coalesce(
                    F.datediff(F.to_date(F.lit("2025-11-11")), F.col("last_baseline_purchase_date")),
                    F.lit(366),
                ),
            )
            .withColumn(
                "baseline_purchase_tenure_days",
                F.coalesce(F.datediff(F.col("last_baseline_purchase_date"), F.col("first_baseline_purchase_date")), F.lit(0)),
            )
            .withColumn(
                "has_baseline_purchase",
                F.when(F.col("last_baseline_purchase_date").isNotNull(), F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()),
            )
            .withColumn(
                "baseline_12m_avg_order_value",
                F.when(F.col("baseline_12m_orders") > 0, F.col("baseline_12m_revenue_sum") / F.col("baseline_12m_orders"))
                .otherwise(F.lit(0.0)),
            )
            .withColumn(
                "baseline_12m_avg_items_per_order",
                F.when(F.col("baseline_12m_orders") > 0, F.col("baseline_12m_quantity_sum") / F.col("baseline_12m_orders"))
                .otherwise(F.lit(0.0)),
            )
            .withColumn(
                "recent_60d_revenue_share",
                F.when(F.col("baseline_12m_revenue_sum") > 0, F.col("baseline_60d_revenue") / F.col("baseline_12m_revenue_sum"))
                .otherwise(F.lit(0.0)),
            )
            .withColumn(
                "recent_60d_order_share",
                F.when(F.col("baseline_12m_orders") > 0, F.col("baseline_60d_orders") / F.col("baseline_12m_orders"))
                .otherwise(F.lit(0.0)),
            )
            .withColumn(
                "campaign_product_revenue_share",
                F.when(
                    F.col("baseline_12m_revenue_sum") > 0,
                    F.col("baseline_campaign_product_revenue") / F.col("baseline_12m_revenue_sum"),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "prior_campaign_product_buyer",
                F.when(F.col("baseline_campaign_product_orders") > 0, F.lit(1)).otherwise(F.lit(0)),
            )
            .withColumn("recent_60d_buyer", F.when(F.col("baseline_60d_orders") > 0, F.lit(1)).otherwise(F.lit(0)))
            .withColumn(
                "lapsed_60d_buyer",
                F.when((F.col("baseline_12m_orders") > 0) & (F.col("baseline_60d_orders") == 0), F.lit(1)).otherwise(
                    F.lit(0)
                ),
            )
            .withColumn(
                "baseline_12m_revenue_sum_bin",
                # ── Spend strata for PSM and exact-stratum matching ───────────
                # The upper bound was previously a single "1000_plus" catch-all.
                # Placebo testing revealed a significant pre-existing spend gap
                # ($8.19/hh) between treatment and control even after PSM, traced
                # to "whale" households (annual spend $2,500–$5,500+) being
                # bucketed with casual $1,050/yr shoppers. The four new upper tiers
                # prevent the matching engine from pairing households across
                # meaningfully different spend trajectories. Lower-end granularity
                # is preserved — it matters for distinguishing non-buyers (zero,
                # lt_10) from occasional shoppers (10_to_49, 50_to_99).
                F.when(F.col("baseline_12m_revenue_sum") == 0, F.lit("zero"))
                .when(F.col("baseline_12m_revenue_sum") < 10, F.lit("lt_10"))
                .when(F.col("baseline_12m_revenue_sum") < 50, F.lit("10_to_49"))
                .when(F.col("baseline_12m_revenue_sum") < 100, F.lit("50_to_99"))
                .when(F.col("baseline_12m_revenue_sum") < 250, F.lit("100_to_249"))
                .when(F.col("baseline_12m_revenue_sum") < 500, F.lit("250_to_499"))
                .when(F.col("baseline_12m_revenue_sum") < 1000, F.lit("500_to_999"))
                .when(F.col("baseline_12m_revenue_sum") < 2000, F.lit("1000_to_1999"))
                .when(F.col("baseline_12m_revenue_sum") < 3500, F.lit("2000_to_3499"))
                .when(F.col("baseline_12m_revenue_sum") < 5000, F.lit("3500_to_4999"))
                .when(F.col("baseline_12m_revenue_sum") < 7500, F.lit("5000_to_7499"))
                .when(F.col("baseline_12m_revenue_sum") < 12000, F.lit("7500_to_11999"))
                .otherwise(F.lit("12000_plus")),
            )
            .withColumn(
                "campaign_product_affinity_label",
                F.when(F.col("baseline_campaign_product_orders") > 1, F.lit("repeat_campaign_product_buyer"))
                .when(F.col("baseline_campaign_product_orders") == 1, F.lit("single_campaign_product_buyer"))
                .otherwise(F.lit("no_prior_campaign_product")),
            )
            .withColumn(
                "baseline_buyer_label",
                F.when(F.col("baseline_60d_orders") > 0, F.lit("recent_buyer"))
                .when((F.col("baseline_12m_orders") > 0) & (F.col("baseline_60d_orders") == 0), F.lit("lapsed_buyer"))
                .otherwise(F.lit("no_12m_purchase")),
            )
            .withColumn(
                "outcome_campaign_product_buyer",
                F.when(F.col("outcome_campaign_product_orders") > 0, F.lit(1)).otherwise(F.lit(0)),
            )
            .drop("first_baseline_purchase_date", "last_baseline_purchase_date")
        )

        # q80 supplies all addressLinks in treatment/control assignment; left joins keep zero-history units.
        assignment_df = exposure_df.select(
            F.col(grain_col),
            F.col("treatment").cast(IntegerType()).alias("treatment"),
            F.col("is_eligible_control").cast(IntegerType()).alias("is_eligible_control"),
            optional_flag_col(exposure_df, "has_partial_exposure_within_addresslink").alias(
                "has_partial_exposure_within_addresslink"
            ),
            optional_col(exposure_df, "min_exposure_ts", F.lit(None).cast(LongType()))
            .cast(LongType())
            .alias("min_exposure_ts"),
            optional_col(exposure_df, "exposure_frequency_deduped", F.lit(0))
            .cast(LongType())
            .alias("exposure_frequency_deduped"),
            optional_col(exposure_df, "mapped_online_identity_count", F.lit(0))
            .cast(LongType())
            .alias("mapped_online_identity_count"),
            optional_col(exposure_df, "exposed_online_identity_count", F.lit(0))
            .cast(LongType())
            .alias("exposed_online_identity_count"),
            optional_col(exposure_df, "person_record_count", F.lit(None).cast(LongType()))
            .cast(LongType())
            .alias("assignment_person_record_count"),
            optional_col(exposure_df, "hhpel_count", F.lit(None).cast(LongType()))
            .cast(LongType())
            .alias("assignment_hhpel_count"),
            optional_col(exposure_df, "mapped_online_identity_count", F.lit(None).cast(LongType()))
            .cast(LongType())
            .alias("assignment_online_identity_count"),
        ).where(F.col(grain_col).isNotNull())

        final_df = assignment_df.join(features_df, grain_col, "left").join(demographic_df, grain_col, "left")

        final_df = (
            final_df.withColumn(
                "hhpel_count",
                F.coalesce(F.col("assignment_hhpel_count"), F.lit(0)).cast(LongType()),
            )
            .withColumn(
                "person_record_count",
                F.coalesce(F.col("assignment_person_record_count"), F.lit(0)).cast(LongType()),
            )
            .withColumn(
                "online_identity_count",
                F.coalesce(F.col("assignment_online_identity_count"), F.lit(0)).cast(LongType()),
            )
        )

        count_fill_columns = [
            "has_partial_exposure_within_addresslink",
            "exposure_frequency_deduped",
            "mapped_online_identity_count",
            "exposed_online_identity_count",
            "hhpel_count",
            "person_record_count",
            "online_identity_count",
            *income_count_columns,
            "num_male_in_addresslink",
            "num_female_in_addresslink",
            "num_unknown_or_other_gender_in_addresslink",
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
            "baseline_12m_orders",
            "baseline_60d_orders",
            "baseline_campaign_product_orders",
            "baseline_12m_negative_transaction_count",
            "baseline_12m_distinct_banners",
            "baseline_12m_distinct_divisions",
            "baseline_12m_active_purchase_days",
            "num_weeks_purchased_in_last_365_days",
            "baseline_purchase_tenure_days",
            "has_baseline_purchase",
            "prior_campaign_product_buyer",
            "recent_60d_buyer",
            "lapsed_60d_buyer",
            "outcome_campaign_product_orders",
            "outcome_campaign_product_buyer",
        ]
        double_fill_columns = [
            "baseline_12m_revenue_sum",
            "baseline_12m_quantity_sum",
            "baseline_60d_revenue",
            "baseline_60d_quantity",
            "baseline_campaign_product_revenue",
            "baseline_campaign_product_quantity",
            "baseline_12m_avg_order_value",
            "baseline_12m_avg_items_per_order",
            "recent_60d_revenue_share",
            "recent_60d_order_share",
            "campaign_product_revenue_share",
            "outcome_campaign_product_revenue",
            "outcome_campaign_product_quantity",
        ]
        string_fill_columns = [
            "poc_label",
            "hh_income_code_profile_label",
            "state_label",
            "age_bucket_profile_label",
            "baseline_12m_revenue_sum_bin",
            "campaign_product_affinity_label",
            "baseline_buyer_label",
        ]

        for col_name in count_fill_columns:
            final_df = final_df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0)).cast(LongType()))
        for col_name in double_fill_columns:
            final_df = final_df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0.0)).cast(DoubleType()))
        for col_name in string_fill_columns:
            final_df = final_df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit("missing")).cast(StringType()))
        final_df = final_df.withColumn(
            "days_since_last_baseline_purchase",
            F.coalesce(F.col("days_since_last_baseline_purchase"), F.lit(366)).cast(LongType()),
        )

        output_cols = [
            F.col(grain_col).cast(StringType()).alias("addressLink"),
            F.col("treatment").cast(IntegerType()).alias("treatment"),
            F.col("is_eligible_control").cast(IntegerType()).alias("is_eligible_control"),
            F.col("has_partial_exposure_within_addresslink")
            .cast(IntegerType())
            .alias("has_partial_exposure_within_addresslink"),
            F.col("min_exposure_ts").cast(LongType()),
            F.col("exposure_frequency_deduped").cast(LongType()),
            F.col("mapped_online_identity_count").cast(LongType()),
            F.col("exposed_online_identity_count").cast(LongType()),
            F.col("hhpel_count").cast(LongType()),
            F.col("person_record_count").cast(LongType()),
            F.col("online_identity_count").cast(LongType()),
            F.col("poc_label").cast(StringType()),
            F.col("hh_income_code_profile_label").cast(StringType()),
            F.col("num_hh_income_missing_in_addresslink").cast(LongType()),
            F.col("num_hh_income_zero_or_negative_in_addresslink").cast(LongType()),
            *[F.col(f"num_hh_income_code_{code}_in_addresslink").cast(LongType()) for code in range(1, 36)],
            F.col("num_hh_income_code_other_in_addresslink").cast(LongType()),
            F.col("state_label").cast(StringType()),
            F.col("num_male_in_addresslink").cast(LongType()),
            F.col("num_female_in_addresslink").cast(LongType()),
            F.col("num_unknown_or_other_gender_in_addresslink").cast(LongType()),
            F.col("age_bucket_profile_label").cast(StringType()),
            F.col("age_missing_count").cast(LongType()),
            F.col("age_lt_18_count").cast(LongType()),
            F.col("age_18_24_count").cast(LongType()),
            F.col("age_25_34_count").cast(LongType()),
            F.col("age_35_44_count").cast(LongType()),
            F.col("age_45_54_count").cast(LongType()),
            F.col("age_55_64_count").cast(LongType()),
            F.col("age_65_74_count").cast(LongType()),
            F.col("age_75_84_count").cast(LongType()),
            F.col("age_85_plus_count").cast(LongType()),
            F.col("baseline_12m_orders").cast(LongType()),
            F.col("baseline_12m_revenue_sum").cast(DoubleType()),
            F.col("baseline_12m_quantity_sum").cast(DoubleType()),
            F.col("baseline_60d_orders").cast(LongType()),
            F.col("baseline_60d_revenue").cast(DoubleType()),
            F.col("baseline_60d_quantity").cast(DoubleType()),
            F.col("baseline_campaign_product_orders").cast(LongType()),
            F.col("baseline_campaign_product_revenue").cast(DoubleType()),
            F.col("baseline_campaign_product_quantity").cast(DoubleType()),
            F.col("baseline_12m_negative_transaction_count").cast(LongType()),
            F.col("baseline_12m_distinct_banners").cast(LongType()),
            F.col("baseline_12m_distinct_divisions").cast(LongType()),
            F.col("baseline_12m_active_purchase_days").cast(LongType()),
            F.col("num_weeks_purchased_in_last_365_days").cast(LongType()),
            F.col("has_baseline_purchase").cast(IntegerType()),
            F.col("days_since_last_baseline_purchase").cast(LongType()),
            F.col("baseline_purchase_tenure_days").cast(LongType()),
            F.col("baseline_12m_avg_order_value").cast(DoubleType()),
            F.col("baseline_12m_avg_items_per_order").cast(DoubleType()),
            F.col("recent_60d_revenue_share").cast(DoubleType()),
            F.col("recent_60d_order_share").cast(DoubleType()),
            F.col("campaign_product_revenue_share").cast(DoubleType()),
            F.col("prior_campaign_product_buyer").cast(IntegerType()),
            F.col("recent_60d_buyer").cast(IntegerType()),
            F.col("lapsed_60d_buyer").cast(IntegerType()),
            F.col("baseline_12m_revenue_sum_bin").cast(StringType()),
            F.col("campaign_product_affinity_label").cast(StringType()),
            F.col("baseline_buyer_label").cast(StringType()),
            F.col("outcome_campaign_product_orders").cast(LongType()),
            F.col("outcome_campaign_product_revenue").cast(DoubleType()),
            F.col("outcome_campaign_product_quantity").cast(DoubleType()),
            F.col("outcome_campaign_product_buyer").cast(IntegerType()),
        ]

        return final_df.select(*output_cols).orderBy(F.desc("treatment"), F.col("addressLink"))
