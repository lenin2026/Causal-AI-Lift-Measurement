import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import DoubleType, IntegerType, StringType

class CustomCode:

    def __init__(self, custom_packages_path=''):
        # The root directory where the custom artifact is installed.
        self.custom_packages_path = custom_packages_path

    def custom_func(
        self,
        spark: SparkSession,
        conversion_df: DataFrame,
        exposure_df: DataFrame,
        demographic_df: DataFrame,
    ) -> DataFrame:

        demographic_df = demographic_df.withColumn("rn", F.row_number().over(Window.partitionBy("addressLink").orderBy(F.desc("install_date"))))\
                                       .filter(F.col("rn") == 1)\
                                       .drop("rn")\
                                       .select("addressLink", "gender", "age", "state", "hh_income")
        
        conversion_df = conversion_df.withColumn('flight_start_unix', F.unix_seconds(F.to_timestamp(F.col("flight_start"))))
        conversion_df = conversion_df.withColumn('flight_end_unix', F.unix_seconds(F.to_timestamp(F.col("flight_end"))))
        conversion_df = conversion_df.withColumn('lookback_start_unix', F.unix_seconds(F.date_sub(F.col("flight_start"), 60).cast("timestamp")))\

        flight_start_unix = conversion_df.select("flight_start_unix").first()[0]

        pre_campaign_w = ((F.col("transaction_timestamp_unix") >= F.col("lookback_start_unix")) & \
                          (F.col("transaction_timestamp_unix") < F.col("flight_start_unix")))
        post_campaign_w = ((F.col("transaction_timestamp_unix") >= F.col("flight_start_unix")) & \
                           (F.col("transaction_timestamp_unix") <= F.col("flight_end_unix")))

        features_df = conversion_df.groupBy("addressLink")\
                                   .agg(F.sum(F.when(pre_campaign_w, F.col("transaction_amount")).otherwise(0)).alias("pre_campaign_total_order_value"), \
                                        F.sum(F.when(pre_campaign_w, F.col("quantity")).otherwise(0)).alias("pre_campaign_total_quantity"), \
                                        F.sum(F.when(pre_campaign_w, F.lit(1)).otherwise(F.lit(0))).alias("pre_campaign_conversion_count"), \
                                        F.max(F.when(pre_campaign_w, F.col("transaction_timestamp_unix"))).alias("pre_campaign_last_txn_time"), \
                                        F.sum(F.when(post_campaign_w, F.col("transaction_amount")).otherwise(0)).alias("post_campaign_total_order_value"), \
                                        F.min(F.when(post_campaign_w, F.col("transaction_timestamp_unix"))).alias("post_campaign_first_seen"), \
                                        F.min(F.col("transaction_timestamp_unix")).alias("first_seen"), \
                                        F.first(F.col("flight_start_unix")).alias("flight_start_unix"))

        features_df = features_df.withColumn('pre_campaign_tenure_days', (F.col("flight_start_unix") - F.col("first_seen")) / F.lit(24 * 60 * 60))\
                                 .withColumn('pre_campaign_conversion_recency', (F.col("flight_start_unix") - F.col("pre_campaign_last_txn_time")) / F.lit(24 * 60 * 60))\
                                 .withColumn('pre_campaign_avg_order_value', F.col("pre_campaign_total_order_value") / F.when(F.col("pre_campaign_conversion_count") > 0, \
                                                                                                                              F.col("pre_campaign_conversion_count")).otherwise(1))\
                                 .withColumn('pre_campaign_avg_items_per_order', F.col("pre_campaign_total_quantity") / F.when(F.col("pre_campaign_conversion_count") > 0, \
                                                                                                                               F.col("pre_campaign_conversion_count")).otherwise(1))\
                                 .withColumn('pre_campaign_has_conversion', F.when(F.col("pre_campaign_conversion_count") > 0, F.lit(1)).otherwise(F.lit(0)))\
                                 .fillna(0).fillna(999, subset=["pre_campaign_conversion_recency"])

        features_df = features_df.join(demographic_df, "addressLink", "left")

        # Treatment = 1 if user was exposed during campaign, and first exposure happened before first conversion in the campaign window
        final_df = features_df.join(exposure_df, "addressLink", "left")
        final_df = final_df.withColumn("treatment", F.when((F.col("min_exposure_ts").isNotNull()) & \
                                                           (F.col("post_campaign_first_seen").isNull() | \
                                                            (F.col("min_exposure_ts") < F.col("post_campaign_first_seen"))), 
                                                            F.lit(1)).otherwise(F.lit(0)))

        final_df = final_df.withColumn("days_into_campaign_at_exposure", F.when(F.col("min_exposure_ts").isNotNull(), \
                                                                                (F.col("min_exposure_ts") - F.lit(flight_start_unix)) / F.lit(24 * 60 * 60))\
                                                                          .otherwise(F.lit(0.0)))
        
        
        q2_users = features_df.select("addressLink")
        q2_df = exposure_df.filter(F.col("is_campaign_exposed") == 1)\
                           .join(q2_users, "addressLink", "left_anti")\
                           .join(features_df.drop("post_campaign_first_seen"), "addressLink", "left")\
                           .withColumn("post_campaign_total_order_value", F.lit(0))\
                           .withColumn("treatment", F.lit(1))\
                           .withColumn("days_into_campaign_at_exposure", (F.col("min_exposure_ts") - F.lit(flight_start_unix)) / F.lit(24 * 60 * 60))\
                           .fillna(0).fillna(999, subset=["pre_campaign_conversion_recency"])
        
        q4_users = exposure_df.filter(F.col("is_campaign_exposed") == 0)\
                              .filter(F.col("is_pre_campaign_exposed") == 1)\
                              .join(final_df.select("addressLink"), "addressLink", "left_anti")\
                              .join(q2_df.select("addressLink"), "addressLink", "left_anti")\
                              .join(demographic_df, "addressLink", "left")
        # Build Q4 DataFrame with same schema as final_df
        # All product features are zero — these users have no product purchase history
        q4_df = q4_users.withColumn("pre_campaign_total_order_value",    F.lit(0.0))\
                        .withColumn("post_campaign_total_order_value",   F.lit(0.0))\
                        .withColumn("treatment",                         F.lit(0))\
                        .withColumn("pre_campaign_tenure_days",          F.lit(0.0))\
                        .withColumn("pre_campaign_conversion_recency",   F.lit(999.0))\
                        .withColumn("pre_campaign_avg_order_value",      F.lit(0.0))\
                        .withColumn("pre_campaign_avg_items_per_order",  F.lit(0.0))\
                        .withColumn("pre_campaign_conversion_count",     F.lit(0))\
                        .withColumn("pre_campaign_has_conversion",       F.lit(0))\
                        .withColumn("exposure_frequency",                F.lit(0))\
                        .withColumn("days_into_campaign_at_exposure",    F.lit(0.0))\
                        .withColumn("min_exposure_ts",                   F.lit(None).cast("long"))\
                        .withColumn("post_campaign_first_seen",          F.lit(None).cast("long"))

        output_cols = [F.col("addressLink"), 
                       F.col("pre_campaign_total_order_value").cast(DoubleType()), 
                       F.col("post_campaign_total_order_value").cast(DoubleType()), 
                       F.col("treatment").cast(IntegerType()), 
                       F.col("pre_campaign_tenure_days").cast(DoubleType()),
                       F.col("pre_campaign_conversion_recency").cast(DoubleType()),
                       F.col("pre_campaign_avg_order_value").cast(DoubleType()),
                       F.col("pre_campaign_avg_items_per_order").cast(DoubleType()),
                       F.col("pre_campaign_conversion_count").cast(IntegerType()),
                       F.col("pre_campaign_has_conversion").cast(IntegerType()),
                       F.col("exposure_frequency").cast(IntegerType()),
                       F.col("days_into_campaign_at_exposure").cast(DoubleType()),
                       F.col("device_platform").cast(StringType()),
                       F.col("impression_device").cast(StringType()),
                       F.col("placement_type").cast(StringType()),
                       F.col("gender").cast(StringType()),
                       F.col("age").cast(IntegerType()),
                       F.col("state").cast(StringType()),
                       F.col("hh_income").cast(IntegerType())]
        string_fill = ["gender", "state", "device_platform", "impression_device", "placement_type"]

        final_df = final_df.select(*output_cols).fillna(0).fillna("NA", subset=string_fill)
        q2_df = q2_df.select(*output_cols).fillna(0).fillna("NA", subset=string_fill)
        q4_df = q4_df.select(*output_cols).fillna(0).fillna("NA", subset=string_fill)
        
        final_df = final_df.union(q2_df).union(q4_df)
        
        return final_df