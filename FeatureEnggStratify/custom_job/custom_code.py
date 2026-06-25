import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType


class CustomCode:
    MODEL_FEATURE_EXCLUDE_COLUMNS = [
        "addressLink",
        "treatment",
        "is_eligible_control",
        "treatment_group",
        "sampled_unit",
        "stratum_rank",
        "treatment_addresslinks",
        "candidate_control_addresslinks",
        "has_partial_exposure_within_addresslink",
        "min_exposure_ts",
        "exposure_frequency_deduped",
        "mapped_online_identity_count",
        "exposed_online_identity_count",
        "hhpel_count",
        "person_record_count",
        "online_identity_count",
        "outcome_campaign_product_orders",
        "outcome_campaign_product_revenue",
        "outcome_campaign_product_quantity",
        "outcome_campaign_product_buyer",
    ]

    def __init__(self, custom_packages_path: str = ""):
        # The root directory where the custom artifact is installed.
        self.custom_packages_path = custom_packages_path

    def custom_func(
        self,
        spark: SparkSession,
        conversion_df: DataFrame,
    ) -> DataFrame:
        """Apply q84B deterministic 1-to-1 stratified sampling.

        In the q84B pipeline step, conversion_df is expected to be q82A output:
        one row per addressLink with treatment/control assignment, stratum
        labels, baseline behavior, and model features.
        """

        grain_col = "addressLink"
        partial_col = "has_partial_exposure_within_addresslink"
        stratum_columns = [
            "poc_label",
            "state_label",
            "baseline_buyer_label",
            "baseline_12m_revenue_sum_bin",
        ]
        q82a_handoff_required_columns = [
            grain_col,
            "treatment",
            "is_eligible_control",
            partial_col,
            "min_exposure_ts",
            "exposure_frequency_deduped",
            "mapped_online_identity_count",
            "exposed_online_identity_count",
            "hhpel_count",
            "person_record_count",
            "online_identity_count",
            "hh_income_code_profile_label",
            "baseline_12m_orders",
            "baseline_12m_revenue_sum",
            "baseline_12m_quantity_sum",
            "outcome_campaign_product_orders",
            "outcome_campaign_product_revenue",
            "outcome_campaign_product_quantity",
            "outcome_campaign_product_buyer",
            *stratum_columns,
        ]
        q82a_optional_columns = [
            "num_hh_income_missing_in_addresslink",
            "num_hh_income_zero_or_negative_in_addresslink",
            *[f"num_hh_income_code_{code}_in_addresslink" for code in range(1, 36)],
            "num_hh_income_code_other_in_addresslink",
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
            "baseline_60d_orders",
            "baseline_60d_revenue",
            "baseline_60d_quantity",
            "baseline_campaign_product_orders",
            "baseline_campaign_product_revenue",
            "baseline_campaign_product_quantity",
            "baseline_12m_negative_transaction_count",
            "baseline_12m_distinct_banners",
            "baseline_12m_distinct_divisions",
            "baseline_12m_active_purchase_days",
            "num_weeks_purchased_in_last_365_days",
            "has_baseline_purchase",
            "days_since_last_baseline_purchase",
            "baseline_purchase_tenure_days",
            "baseline_12m_avg_order_value",
            "baseline_12m_avg_items_per_order",
            "recent_60d_revenue_share",
            "recent_60d_order_share",
            "campaign_product_revenue_share",
            "prior_campaign_product_buyer",
            "recent_60d_buyer",
            "lapsed_60d_buyer",
            "campaign_product_affinity_label",
        ]

        def require_columns(df: DataFrame, required_columns, df_name: str) -> None:
            missing_columns = [col_name for col_name in required_columns if col_name not in df.columns]
            if missing_columns:
                available_columns = ", ".join(df.columns[:40])
                raise ValueError(
                    f"{df_name} missing required columns: {', '.join(missing_columns)}. "
                    f"For q84B, conversion_df must be the q82A addresslink_features_with_assignment output. "
                    f"Available columns: {available_columns}"
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

        q82a_addresslink_features_with_assignment = canonicalize_columns(
            conversion_df,
            q82a_handoff_required_columns + q82a_optional_columns,
        )

        # Backward compatibility: map pre-1.3 q82A column names to current names.
        # Applies only when the old name is present and the new name is absent,
        # so it is safe to leave in place after AllFeatures is regenerated with the 1.3 wheel.
        _legacy_renames = {
            "baseline_12m_revenue":     "baseline_12m_revenue_sum",
            "baseline_12m_quantity":    "baseline_12m_quantity_sum",
            "baseline_12m_revenue_bin": "baseline_12m_revenue_sum_bin",
        }
        for _old, _new in _legacy_renames.items():
            if _old in q82a_addresslink_features_with_assignment.columns and \
               _new not in q82a_addresslink_features_with_assignment.columns:
                q82a_addresslink_features_with_assignment = q82a_addresslink_features_with_assignment.withColumnRenamed(_old, _new)

        require_columns(
            q82a_addresslink_features_with_assignment,
            q82a_handoff_required_columns,
            "conversion_df must be q82A addresslink_features_with_assignment",
        )

        final_df = q82a_addresslink_features_with_assignment

        final_df = final_df.withColumn(grain_col, F.trim(F.col(grain_col).cast(StringType())))
        for col_name in stratum_columns:
            final_df = final_df.withColumn(
                col_name,
                F.coalesce(F.trim(F.col(col_name).cast(StringType())), F.lit("missing")),
            )

        match_ready = (
            final_df.filter(F.col(grain_col).isNotNull() & (F.col(grain_col) != ""))
            .filter(
                (F.col("treatment").cast(IntegerType()) == 1)
                | (
                    (F.col("treatment").cast(IntegerType()) == 0)
                    & (F.col("is_eligible_control").cast(IntegerType()) == 1)
                    & (F.coalesce(F.col(partial_col).cast(IntegerType()), F.lit(0)) == 0)
                )
            )
            .withColumn(
                "treatment_group",
                F.when(F.col("treatment").cast(IntegerType()) == 1, F.lit("treatment")).otherwise(
                    F.lit("candidate_control")
                ),
            )
        )

        stratum_counts = (
            match_ready.groupBy(*stratum_columns)
            .agg(
                F.sum(F.when(F.col("treatment_group") == "treatment", 1).otherwise(0))
                .cast(LongType())
                .alias("treatment_addresslinks"),
                F.sum(F.when(F.col("treatment_group") == "candidate_control", 1).otherwise(0))
                .cast(LongType())
                .alias("candidate_control_addresslinks"),
            )
            .filter(F.col("treatment_addresslinks") > 0)
        )

        final_df = (
            match_ready.alias("a")
            .join(stratum_counts.alias("s"), on=stratum_columns, how="inner")
            .withColumn(
                "stratum_rank",
                F.row_number().over(
                    Window.partitionBy(*[F.col(col_name) for col_name in stratum_columns], F.col("treatment_group"))
                    .orderBy(F.xxhash64(F.col(grain_col).cast(StringType())), F.col(grain_col))
                ),
            )
            .filter(
                (F.col("treatment_group") == "treatment")
                | (
                    (F.col("treatment_group") == "candidate_control")
                    & (F.col("stratum_rank") <= F.col("treatment_addresslinks"))
                )
            )
            .withColumn("sampled_unit", F.lit(1).cast(IntegerType()))
        )

        output_cols = [
            # addressLink: current household analysis key from q82A; used to join sampled units back to features.
            F.col(grain_col).cast(StringType()).alias("addressLink"),
            # treatment: q80A/q82A exposure assignment; 1 if any mapped online identity under addressLink was exposed.
            F.col("treatment").cast(IntegerType()),
            # is_eligible_control: q80A/q82A flag; 1 only when addressLink has no campaign-period exposure.
            F.col("is_eligible_control").cast(IntegerType()),
            # treatment_group: q84B label derived from treatment; values are treatment or candidate_control.
            F.col("treatment_group").cast(StringType()),
            # sampled_unit: q84B flag; 1 for every row retained after deterministic stratified sampling.
            F.col("sampled_unit").cast(IntegerType()),
            # stratum_rank: q84B deterministic row number within exact stratum and treatment_group using xxhash64(addressLink).
            F.col("stratum_rank").cast(LongType()),
            # treatment_addresslinks: q84B count of treated addressLinks in the exact stratum.
            F.col("treatment_addresslinks").cast(LongType()),
            # candidate_control_addresslinks: q84B count of eligible candidate controls in the exact stratum before 1:1 sampling.
            F.col("candidate_control_addresslinks").cast(LongType()),
            # has_partial_exposure_within_addresslink: q80A/q82A flag; 1 when same addressLink has exposed and unexposed online identities.
            F.col(partial_col).cast(IntegerType()).alias("has_partial_exposure_within_addresslink"),
            # min_exposure_ts: earliest deduped campaign exposure timestamp rolled up to addressLink in q80A/q82A.
            optional_col(final_df, "min_exposure_ts", F.lit(None)).cast(LongType()).alias("min_exposure_ts"),
            # exposure_frequency_deduped: deduped campaign exposure event count rolled up to addressLink in q80A/q82A.
            optional_col(final_df, "exposure_frequency_deduped", F.lit(None))
            .cast(LongType())
            .alias("exposure_frequency_deduped"),
            # mapped_online_identity_count: q80A/q82A count of mapped online identities under addressLink.
            optional_col(final_df, "mapped_online_identity_count", F.lit(None))
            .cast(LongType())
            .alias("mapped_online_identity_count"),
            # exposed_online_identity_count: q80A/q82A count of mapped online identities under addressLink with exposure.
            optional_col(final_df, "exposed_online_identity_count", F.lit(None))
            .cast(LongType())
            .alias("exposed_online_identity_count"),
            # hhpel_count: q80A/q82A count of distinct hhpel values under addressLink for diagnostics.
            optional_col(final_df, "hhpel_count", F.lit(None)).cast(LongType()).alias("hhpel_count"),
            # person_record_count: q80A/q82A count of distinct Grouping_Indicator records under addressLink.
            optional_col(final_df, "person_record_count", F.lit(None)).cast(LongType()).alias("person_record_count"),
            # online_identity_count: q82A assignment count of online identity IDs under addressLink.
            optional_col(final_df, "online_identity_count", F.lit(None)).cast(LongType()).alias("online_identity_count"),
            # poc_label: q82A child-presence label from latest demographics rolled up to addressLink.
            F.col("poc_label").cast(StringType()),
            # hh_income_code_profile_label: q82A compact profile of hh_income codes under addressLink for balance review.
            optional_col(final_df, "hh_income_code_profile_label", F.lit("missing"))
            .cast(StringType())
            .alias("hh_income_code_profile_label"),
            # num_hh_income_missing_in_addresslink: q82A count of latest demographic records with missing hh_income.
            optional_col(final_df, "num_hh_income_missing_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_missing_in_addresslink"),
            # num_hh_income_zero_or_negative_in_addresslink: q82A count of latest demographic records with hh_income <= 0.
            optional_col(final_df, "num_hh_income_zero_or_negative_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_zero_or_negative_in_addresslink"),
            # num_hh_income_code_1_in_addresslink: q82A count of latest demographic records with hh_income code 1.
            optional_col(final_df, "num_hh_income_code_1_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_1_in_addresslink"),
            # num_hh_income_code_2_in_addresslink: q82A count of latest demographic records with hh_income code 2.
            optional_col(final_df, "num_hh_income_code_2_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_2_in_addresslink"),
            # num_hh_income_code_3_in_addresslink: q82A count of latest demographic records with hh_income code 3.
            optional_col(final_df, "num_hh_income_code_3_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_3_in_addresslink"),
            # num_hh_income_code_4_in_addresslink: q82A count of latest demographic records with hh_income code 4.
            optional_col(final_df, "num_hh_income_code_4_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_4_in_addresslink"),
            # num_hh_income_code_5_in_addresslink: q82A count of latest demographic records with hh_income code 5.
            optional_col(final_df, "num_hh_income_code_5_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_5_in_addresslink"),
            # num_hh_income_code_6_in_addresslink: q82A count of latest demographic records with hh_income code 6.
            optional_col(final_df, "num_hh_income_code_6_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_6_in_addresslink"),
            # num_hh_income_code_7_in_addresslink: q82A count of latest demographic records with hh_income code 7.
            optional_col(final_df, "num_hh_income_code_7_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_7_in_addresslink"),
            # num_hh_income_code_8_in_addresslink: q82A count of latest demographic records with hh_income code 8.
            optional_col(final_df, "num_hh_income_code_8_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_8_in_addresslink"),
            # num_hh_income_code_9_in_addresslink: q82A count of latest demographic records with hh_income code 9.
            optional_col(final_df, "num_hh_income_code_9_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_9_in_addresslink"),
            # num_hh_income_code_10_in_addresslink: q82A count of latest demographic records with hh_income code 10.
            optional_col(final_df, "num_hh_income_code_10_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_10_in_addresslink"),
            # num_hh_income_code_11_in_addresslink: q82A count of latest demographic records with hh_income code 11.
            optional_col(final_df, "num_hh_income_code_11_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_11_in_addresslink"),
            # num_hh_income_code_12_in_addresslink: q82A count of latest demographic records with hh_income code 12.
            optional_col(final_df, "num_hh_income_code_12_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_12_in_addresslink"),
            # num_hh_income_code_13_in_addresslink: q82A count of latest demographic records with hh_income code 13.
            optional_col(final_df, "num_hh_income_code_13_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_13_in_addresslink"),
            # num_hh_income_code_14_in_addresslink: q82A count of latest demographic records with hh_income code 14.
            optional_col(final_df, "num_hh_income_code_14_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_14_in_addresslink"),
            # num_hh_income_code_15_in_addresslink: q82A count of latest demographic records with hh_income code 15.
            optional_col(final_df, "num_hh_income_code_15_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_15_in_addresslink"),
            # num_hh_income_code_16_in_addresslink: q82A count of latest demographic records with hh_income code 16.
            optional_col(final_df, "num_hh_income_code_16_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_16_in_addresslink"),
            # num_hh_income_code_17_in_addresslink: q82A count of latest demographic records with hh_income code 17.
            optional_col(final_df, "num_hh_income_code_17_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_17_in_addresslink"),
            # num_hh_income_code_18_in_addresslink: q82A count of latest demographic records with hh_income code 18.
            optional_col(final_df, "num_hh_income_code_18_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_18_in_addresslink"),
            # num_hh_income_code_19_in_addresslink: q82A count of latest demographic records with hh_income code 19.
            optional_col(final_df, "num_hh_income_code_19_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_19_in_addresslink"),
            # num_hh_income_code_20_in_addresslink: q82A count of latest demographic records with hh_income code 20.
            optional_col(final_df, "num_hh_income_code_20_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_20_in_addresslink"),
            # num_hh_income_code_21_in_addresslink: q82A count of latest demographic records with hh_income code 21.
            optional_col(final_df, "num_hh_income_code_21_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_21_in_addresslink"),
            # num_hh_income_code_22_in_addresslink: q82A count of latest demographic records with hh_income code 22.
            optional_col(final_df, "num_hh_income_code_22_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_22_in_addresslink"),
            # num_hh_income_code_23_in_addresslink: q82A count of latest demographic records with hh_income code 23.
            optional_col(final_df, "num_hh_income_code_23_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_23_in_addresslink"),
            # num_hh_income_code_24_in_addresslink: q82A count of latest demographic records with hh_income code 24.
            optional_col(final_df, "num_hh_income_code_24_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_24_in_addresslink"),
            # num_hh_income_code_25_in_addresslink: q82A count of latest demographic records with hh_income code 25.
            optional_col(final_df, "num_hh_income_code_25_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_25_in_addresslink"),
            # num_hh_income_code_26_in_addresslink: q82A count of latest demographic records with hh_income code 26.
            optional_col(final_df, "num_hh_income_code_26_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_26_in_addresslink"),
            # num_hh_income_code_27_in_addresslink: q82A count of latest demographic records with hh_income code 27.
            optional_col(final_df, "num_hh_income_code_27_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_27_in_addresslink"),
            # num_hh_income_code_28_in_addresslink: q82A count of latest demographic records with hh_income code 28.
            optional_col(final_df, "num_hh_income_code_28_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_28_in_addresslink"),
            # num_hh_income_code_29_in_addresslink: q82A count of latest demographic records with hh_income code 29.
            optional_col(final_df, "num_hh_income_code_29_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_29_in_addresslink"),
            # num_hh_income_code_30_in_addresslink: q82A count of latest demographic records with hh_income code 30.
            optional_col(final_df, "num_hh_income_code_30_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_30_in_addresslink"),
            # num_hh_income_code_31_in_addresslink: q82A count of latest demographic records with hh_income code 31.
            optional_col(final_df, "num_hh_income_code_31_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_31_in_addresslink"),
            # num_hh_income_code_32_in_addresslink: q82A count of latest demographic records with hh_income code 32.
            optional_col(final_df, "num_hh_income_code_32_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_32_in_addresslink"),
            # num_hh_income_code_33_in_addresslink: q82A count of latest demographic records with hh_income code 33.
            optional_col(final_df, "num_hh_income_code_33_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_33_in_addresslink"),
            # num_hh_income_code_34_in_addresslink: q82A count of latest demographic records with hh_income code 34.
            optional_col(final_df, "num_hh_income_code_34_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_34_in_addresslink"),
            # num_hh_income_code_35_in_addresslink: q82A count of latest demographic records with hh_income code 35.
            optional_col(final_df, "num_hh_income_code_35_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_35_in_addresslink"),
            # num_hh_income_code_other_in_addresslink: q82A count of latest demographic records with positive hh_income outside 1..35.
            optional_col(final_df, "num_hh_income_code_other_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_hh_income_code_other_in_addresslink"),
            # state_label: q82A state rollup from latest demographics; missing or mixed values are labeled.
            F.col("state_label").cast(StringType()),
            # num_male_in_addresslink: q82A count of latest demographic records under addressLink with male gender.
            optional_col(final_df, "num_male_in_addresslink", F.lit(0)).cast(LongType()).alias("num_male_in_addresslink"),
            # num_female_in_addresslink: q82A count of latest demographic records under addressLink with female gender.
            optional_col(final_df, "num_female_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_female_in_addresslink"),
            # num_unknown_or_other_gender_in_addresslink: q82A count of latest demographic records with missing/other gender.
            optional_col(final_df, "num_unknown_or_other_gender_in_addresslink", F.lit(0))
            .cast(LongType())
            .alias("num_unknown_or_other_gender_in_addresslink"),
            # age_bucket_profile_label: q82A pipe-delimited age bucket profile under addressLink from latest demographics.
            optional_col(final_df, "age_bucket_profile_label", F.lit("missing"))
            .cast(StringType())
            .alias("age_bucket_profile_label"),
            # age_missing_count: q82A count of latest demographic records under addressLink with missing age.
            optional_col(final_df, "age_missing_count", F.lit(0)).cast(LongType()).alias("age_missing_count"),
            # age_lt_18_count: q82A count of latest demographic records under addressLink with age < 18.
            optional_col(final_df, "age_lt_18_count", F.lit(0)).cast(LongType()).alias("age_lt_18_count"),
            # age_18_24_count: q82A count of latest demographic records under addressLink with age 18..24.
            optional_col(final_df, "age_18_24_count", F.lit(0)).cast(LongType()).alias("age_18_24_count"),
            # age_25_34_count: q82A count of latest demographic records under addressLink with age 25..34.
            optional_col(final_df, "age_25_34_count", F.lit(0)).cast(LongType()).alias("age_25_34_count"),
            # age_35_44_count: q82A count of latest demographic records under addressLink with age 35..44.
            optional_col(final_df, "age_35_44_count", F.lit(0)).cast(LongType()).alias("age_35_44_count"),
            # age_45_54_count: q82A count of latest demographic records under addressLink with age 45..54.
            optional_col(final_df, "age_45_54_count", F.lit(0)).cast(LongType()).alias("age_45_54_count"),
            # age_55_64_count: q82A count of latest demographic records under addressLink with age 55..64.
            optional_col(final_df, "age_55_64_count", F.lit(0)).cast(LongType()).alias("age_55_64_count"),
            # age_65_74_count: q82A count of latest demographic records under addressLink with age 65..74.
            optional_col(final_df, "age_65_74_count", F.lit(0)).cast(LongType()).alias("age_65_74_count"),
            # age_75_84_count: q82A count of latest demographic records under addressLink with age 75..84.
            optional_col(final_df, "age_75_84_count", F.lit(0)).cast(LongType()).alias("age_75_84_count"),
            # age_85_plus_count: q82A count of latest demographic records under addressLink with age >= 85.
            optional_col(final_df, "age_85_plus_count", F.lit(0)).cast(LongType()).alias("age_85_plus_count"),
            # baseline_12m_orders: q82A distinct order count during 2024-11-12 through 2025-11-11.
            optional_col(final_df, "baseline_12m_orders", F.lit(0)).cast(LongType()).alias("baseline_12m_orders"),
            # baseline_12m_revenue_sum: q82A positive transaction_amount sum during 2024-11-12 through 2025-11-11.
            optional_col(final_df, "baseline_12m_revenue_sum", F.lit(0.0))
            .cast(DoubleType())
            .alias("baseline_12m_revenue_sum"),
            # baseline_12m_quantity_sum: q82A quantity sum during 2024-11-12 through 2025-11-11.
            optional_col(final_df, "baseline_12m_quantity_sum", F.lit(0.0))
            .cast(DoubleType())
            .alias("baseline_12m_quantity_sum"),
            # baseline_60d_orders: q82A distinct order count during 2025-09-13 through 2025-11-11.
            optional_col(final_df, "baseline_60d_orders", F.lit(0)).cast(LongType()).alias("baseline_60d_orders"),
            # baseline_60d_revenue: q82A positive transaction_amount sum during 2025-09-13 through 2025-11-11.
            optional_col(final_df, "baseline_60d_revenue", F.lit(0.0))
            .cast(DoubleType())
            .alias("baseline_60d_revenue"),
            # baseline_60d_quantity: q82A quantity sum during 2025-09-13 through 2025-11-11.
            optional_col(final_df, "baseline_60d_quantity", F.lit(0.0))
            .cast(DoubleType())
            .alias("baseline_60d_quantity"),
            # baseline_campaign_product_orders: q82A distinct campaign-product orders in the 12-month baseline.
            optional_col(final_df, "baseline_campaign_product_orders", F.lit(0))
            .cast(LongType())
            .alias("baseline_campaign_product_orders"),
            # baseline_campaign_product_revenue: q82A positive campaign-product revenue in the 12-month baseline.
            optional_col(final_df, "baseline_campaign_product_revenue", F.lit(0.0))
            .cast(DoubleType())
            .alias("baseline_campaign_product_revenue"),
            # baseline_campaign_product_quantity: q82A campaign-product quantity in the 12-month baseline.
            optional_col(final_df, "baseline_campaign_product_quantity", F.lit(0.0))
            .cast(DoubleType())
            .alias("baseline_campaign_product_quantity"),
            # baseline_12m_negative_transaction_count: q82A count of negative transaction_amount rows in the 12-month baseline.
            optional_col(final_df, "baseline_12m_negative_transaction_count", F.lit(0))
            .cast(LongType())
            .alias("baseline_12m_negative_transaction_count"),
            # baseline_12m_distinct_banners: q82A count of distinct banner values purchased in the 12-month baseline.
            optional_col(final_df, "baseline_12m_distinct_banners", F.lit(0))
            .cast(LongType())
            .alias("baseline_12m_distinct_banners"),
            # baseline_12m_distinct_divisions: q82A count of distinct division values purchased in the 12-month baseline.
            optional_col(final_df, "baseline_12m_distinct_divisions", F.lit(0))
            .cast(LongType())
            .alias("baseline_12m_distinct_divisions"),
            # baseline_12m_active_purchase_days: q82A count of distinct purchase dates in the 12-month baseline.
            optional_col(final_df, "baseline_12m_active_purchase_days", F.lit(0))
            .cast(LongType())
            .alias("baseline_12m_active_purchase_days"),
            # num_weeks_purchased_in_last_365_days: q82A count of distinct purchase weeks in the 12-month baseline.
            optional_col(final_df, "num_weeks_purchased_in_last_365_days", F.lit(0))
            .cast(LongType())
            .alias("num_weeks_purchased_in_last_365_days"),
            # has_baseline_purchase: q82A flag; 1 when at least one purchase exists in the 12-month baseline, else 0.
            optional_col(final_df, "has_baseline_purchase", F.lit(0))
            .cast(IntegerType())
            .alias("has_baseline_purchase"),
            # days_since_last_baseline_purchase: q82A days from 2025-11-11 to most recent baseline purchase; 366 if none (one day beyond the 12-month window max, regression-safe).
            optional_col(final_df, "days_since_last_baseline_purchase", F.lit(366))
            .cast(LongType())
            .alias("days_since_last_baseline_purchase"),
            # baseline_purchase_tenure_days: q82A days between first and last baseline purchase date; 0 if no purchases (use has_baseline_purchase to distinguish no-purchase from single-purchase).
            optional_col(final_df, "baseline_purchase_tenure_days", F.lit(0))
            .cast(LongType())
            .alias("baseline_purchase_tenure_days"),
            # baseline_12m_avg_order_value: q82A baseline_12m_revenue_sum divided by baseline_12m_orders when orders > 0.
            optional_col(final_df, "baseline_12m_avg_order_value", F.lit(0.0))
            .cast(DoubleType())
            .alias("baseline_12m_avg_order_value"),
            # baseline_12m_avg_items_per_order: q82A baseline_12m_quantity_sum divided by baseline_12m_orders when orders > 0.
            optional_col(final_df, "baseline_12m_avg_items_per_order", F.lit(0.0))
            .cast(DoubleType())
            .alias("baseline_12m_avg_items_per_order"),
            # recent_60d_revenue_share: q82A baseline_60d_revenue divided by baseline_12m_revenue_sum when revenue > 0.
            optional_col(final_df, "recent_60d_revenue_share", F.lit(0.0))
            .cast(DoubleType())
            .alias("recent_60d_revenue_share"),
            # recent_60d_order_share: q82A baseline_60d_orders divided by baseline_12m_orders when orders > 0.
            optional_col(final_df, "recent_60d_order_share", F.lit(0.0))
            .cast(DoubleType())
            .alias("recent_60d_order_share"),
            # campaign_product_revenue_share: q82A baseline campaign-product revenue divided by total baseline revenue.
            optional_col(final_df, "campaign_product_revenue_share", F.lit(0.0))
            .cast(DoubleType())
            .alias("campaign_product_revenue_share"),
            # prior_campaign_product_buyer: q82A flag; 1 when baseline_campaign_product_orders > 0.
            optional_col(final_df, "prior_campaign_product_buyer", F.lit(0))
            .cast(IntegerType())
            .alias("prior_campaign_product_buyer"),
            # recent_60d_buyer: q82A flag; 1 when baseline_60d_orders > 0.
            optional_col(final_df, "recent_60d_buyer", F.lit(0)).cast(IntegerType()).alias("recent_60d_buyer"),
            # lapsed_60d_buyer: q82A flag; 1 when 12-month buyer had no 60-day baseline orders.
            optional_col(final_df, "lapsed_60d_buyer", F.lit(0)).cast(IntegerType()).alias("lapsed_60d_buyer"),
            # baseline_12m_revenue_sum_bin: q82A revenue stratum label derived from baseline_12m_revenue_sum.
            F.col("baseline_12m_revenue_sum_bin").cast(StringType()),
            # campaign_product_affinity_label: q82A label summarizing prior campaign-product buying affinity.
            optional_col(final_df, "campaign_product_affinity_label", F.lit("missing"))
            .cast(StringType())
            .alias("campaign_product_affinity_label"),
            # baseline_buyer_label: q82A buyer stratum label such as recent_buyer, lapsed_buyer, or no_12m_purchase.
            F.col("baseline_buyer_label").cast(StringType()),
            # outcome_campaign_product_orders: q82A campaign-period distinct campaign-product order count; outcome, not covariate.
            optional_col(final_df, "outcome_campaign_product_orders", F.lit(0))
            .cast(LongType())
            .alias("outcome_campaign_product_orders"),
            # outcome_campaign_product_revenue: q82A campaign-period positive campaign-product revenue; outcome, not covariate.
            optional_col(final_df, "outcome_campaign_product_revenue", F.lit(0.0))
            .cast(DoubleType())
            .alias("outcome_campaign_product_revenue"),
            # outcome_campaign_product_quantity: q82A campaign-period campaign-product quantity; outcome, not covariate.
            optional_col(final_df, "outcome_campaign_product_quantity", F.lit(0.0))
            .cast(DoubleType())
            .alias("outcome_campaign_product_quantity"),
            # outcome_campaign_product_buyer: q82A campaign-period buyer flag; outcome, not covariate.
            optional_col(final_df, "outcome_campaign_product_buyer", F.lit(0))
            .cast(IntegerType())
            .alias("outcome_campaign_product_buyer"),
        ]

        return final_df.select(*output_cols).orderBy(
            F.desc("treatment"),
            F.col("poc_label"),
            F.col("state_label"),
            F.col("baseline_buyer_label"),
            F.col("baseline_12m_revenue_sum_bin"),
            F.col("treatment_group").desc(),
            F.col("stratum_rank"),
            F.col("addressLink"),
        )
