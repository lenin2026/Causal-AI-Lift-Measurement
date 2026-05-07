import pyspark.sql.functions as F
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
        map_df: DataFrame,
        campaign_df: DataFrame,
        demographic_df: DataFrame,
        CAMPAIGN_ID: int,
    ) -> DataFrame:

        # campaign_df = campaign_df.join(campaign_map_df, campaign_df["campaign_id"] == campaign_map_df["campaign_id"])
        # campaign = campaign_df.filter(F.col("campaign_id") == CAMPAIGN_ID).limit(1).collect()
        # flight_start = campaign[0]["flight_start"]
        # flight_end = campaign[0]["flight_end"]
        # campaign_start_unix = F.unix_seconds(F.to_timestamp(F.lit(flight_start)))
        # campaign_end_unix = F.unix_seconds(F.to_timestamp(F.lit(flight_end)))
        # lookback_start_unix = F.unix_seconds(F.to_timestamp(F.lit(flight_start - 60)))

        campaign_df = campaign_df.filter(F.col("campaign_id") == CAMPAIGN_ID)
        campaign_df = campaign_df.withColumn('flight_start_unix', F.unix_seconds(F.to_timestamp(F.col("flight_start"))))
        campaign_df = campaign_df.withColumn('flight_end_unix', F.unix_seconds(F.to_timestamp(F.col("flight_end"))))
        campaign_df = campaign_df.withColumn('lookback_start_unix', F.unix_seconds(F.date_sub(F.col("flight_start"), 60).cast("timestamp")))

        campaign_row = campaign_df.collect()[0]
        flight_start_unix = campaign_row["flight_start_unix"]
        flight_end_unix = campaign_row["flight_end_unix"]
        lookback_start_unix = campaign_row["lookback_start_unix"]
    
        mapped_exposure_df = exposure_df.filter(F.col("campaign_id") == CAMPAIGN_ID)\
                                        .filter((F.col("ts") >= F.lit(flight_start_unix)) & \
                                                (F.col("ts") <= F.lit(flight_end_unix)))\
                                        .join(map_df, exposure_df["tp_id"] == map_df["rampid_meta"])
        exposure_agg_df = mapped_exposure_df.select(F.col("rampid_rmn").alias("lr_id"),\
                                                    F.col("ts").alias("exposure_ts"),
                                                    F.col("device_platform"),
                                                    F.col("impression_device"),
                                                    F.col("placement_type"))\
                                            .groupBy("lr_id")\
                                            .agg(F.min("exposure_ts").alias("min_exposure_ts"),
                                                 F.count("*").alias("exposure_frequency"),
                                                 F.first("device_platform").alias("device_platform"),
                                                 F.first("impression_device").alias("impression_device"),
                                                 F.first("placement_type").alias("placement_type"))

        # Q4 - non exposed non converted
        # Users exposed to ANY campaign in lookback but NOT this campaign
        # These are ad-reachable users with no product purchase history
        pre_campaign_exposed = exposure_df.filter((F.col("ts") >= F.lit(lookback_start_unix)) &
                                                  (F.col("ts") <  F.lit(flight_start_unix)))\
                                          .join(map_df, exposure_df["tp_id"] == map_df["rampid_meta"])\
                                          .select(F.col("rampid_rmn").alias("lr_id"),
                                                  F.col("device_platform"),
                                                  F.col("impression_device"),
                                                  F.col("placement_type"))\
                                          .groupBy("lr_id")\
                                          .agg(F.first("device_platform").alias("device_platform"),
                                               F.first("impression_device").alias("impression_device"),
                                               F.first("placement_type").alias("placement_type"))

        conversion_df  = conversion_df.join(campaign_df, conversion_df["product_id"] == campaign_df["product_id"])
        conversion_df = conversion_df.filter((F.col("transaction_timestamp_unix") >= F.col("lookback_start_unix")) & \
                                             (F.col("transaction_timestamp_unix") <= F.col("flight_end_unix")))

        pre_campaign_w = ((F.col("transaction_timestamp_unix") >= F.col("lookback_start_unix")) & \
                          (F.col("transaction_timestamp_unix") < F.col("flight_start_unix")))
        post_campaign_w = ((F.col("transaction_timestamp_unix") >= F.col("flight_start_unix")) & \
                           (F.col("transaction_timestamp_unix") <= F.col("flight_end_unix")))

        features_df = conversion_df.groupBy("lr_id")\
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
                                                                             F.col("pre_campaign_conversion_count")).otherwise(0))\
                                 .withColumn('pre_campaign_avg_items_per_order', F.col("pre_campaign_total_quantity") / F.when(F.col("pre_campaign_conversion_count") > 0, \
                                                                                                                               F.col("pre_campaign_conversion_count")).otherwise(1))\
                                 .withColumn('pre_campaign_has_conversion', F.when(F.col("pre_campaign_conversion_count") > 0, F.lit(1)).otherwise(F.lit(0)))\
                                 .fillna(0).fillna(999, subset=["pre_campaign_conversion_recency"])

        # post_spend_99p, pre_spend_99p = features_df.approxQuantile(["post_campaign_total_order_value", "pre_campaign_total_order_value"], [0.99], 0.01)
        # post_spend_99p = post_spend_99p[0]
        # pre_spend_99p = pre_spend_99p[0]

        # features_df = features_df.withColumn("post_campaign_total_order_value", F.when(F.col("post_campaign_total_order_value") > post_spend_99p, \
        #                                                                                F.lit(post_spend_99p))\
        #                                                                          .otherwise(F.col("post_campaign_total_order_value")))
        # features_df = features_df.withColumn("pre_campaign_total_order_value", F.when(F.col("pre_campaign_total_order_value") > pre_spend_99p, \
        #                                                                               F.lit(pre_spend_99p))\
        #                                                                         .otherwise(F.col("pre_campaign_total_order_value")))

        # Treatment = 1 if user was exposed during campaign, and first exposure happened before first conversion in the campaign window
        final_df = features_df.join(exposure_agg_df, "lr_id", "left")
        final_df = final_df.withColumn("treatment", F.when((F.col("min_exposure_ts").isNotNull()) & \
                                                           (F.col("post_campaign_first_seen").isNull() | \
                                                            (F.col("min_exposure_ts") < F.col("post_campaign_first_seen"))), 
                                                            F.lit(1)).otherwise(F.lit(0)))

        final_df = final_df.withColumn("days_into_campaign_at_exposure", F.when(F.col("min_exposure_ts").isNotNull(), \
                                                                                (F.col("min_exposure_ts") - F.lit(flight_start_unix)) / F.lit(24 * 60 * 60))\
                                                                          .otherwise(F.lit(0.0)))
        
        
        q2_users = features_df.filter(F.col("post_campaign_total_order_value") > 0).select("lr_id")
        q2_df = exposure_agg_df.join(q2_users, "lr_id", "left_anti")\
                               .join(features_df.drop("post_campaign_first_seen"), "lr_id", "left")\
                               .withColumn("treatment", F.lit(1))\
                               .withColumn("days_into_campaign_at_exposure", (F.col("min_exposure_ts") - F.lit(flight_start_unix)) / F.lit(24 * 60 * 60))\
                               .fillna(0).fillna(999, subset=["pre_campaign_conversion_recency"])
        
        q4_users = pre_campaign_exposed.join(final_df.select("lr_id"), pre_campaign_exposed["lr_id"] == final_df["lr_id"], "left_anti")\
                                       .join(q2_df, "lr_id", "left_anti")
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

        final_df = final_df.join(map_df, final_df["lr_id"] == map_df["rampid_rmn"], "left")
        final_df = final_df.join(demographic_df, F.col("rampid_dpm") == demographic_df["rampid"],"left")
        q2_df = q2_df.join(map_df, q2_df["lr_id"] == map_df["rampid_rmn"], "left")
        q2_df = q2_df.join(demographic_df, F.col("rampid_dpm") == demographic_df["rampid"],"left")
        q4_df = q4_df.join(map_df, q4_df["lr_id"] == map_df["rampid_rmn"], "left")
        q4_df = q4_df.join(demographic_df, F.col("rampid_dpm") == demographic_df["rampid"],"left")

        # final_df = final_df.join(demographic_df, final_df["lr_id"] == demographic_df["rampid"], "left")

        output_cols = [F.col("lr_id"), 
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