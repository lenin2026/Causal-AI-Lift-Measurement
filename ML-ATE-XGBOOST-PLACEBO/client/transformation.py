import os

from pyspark.sql import SparkSession, DataFrame

from client.data_handler import DataHandler, BaseTablePy, JoinTablePy
from custom_job.custom_code import CustomCode



class Transformation:

    def __init__(self, spark: SparkSession, has_extra_outputs=False, custom_packages_path='', macro_to_columns=None, runner=None, extra_output_macro_data_location_map=None, *args, **kwargs):
        self.spark = spark
        self.custom_packages_path = custom_packages_path
        self.macro_to_columns = macro_to_columns
        self.data_handler = DataHandler(self.spark, has_extra_outputs, custom_packages_path, macro_to_columns, runner, extra_output_macro_data_location_map, kwargs.get("run_parameters", {}))

    def transform(self):
        psm_matched_features_df = self.data_handler.read("PSMMatchedFeatures")

        # Placebo: replace total campaign-period outcome with pre-campaign 60-day
        # revenue. custom_func still reads "outcome_total_campaign_revenue" but now
        # those values come from the pre-period baseline — where the campaign had
        # not yet launched. A significant τ̂ ≠ 0 signals pre-existing spend imbalance
        # between PSM treatment and control groups.
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType
        psm_matched_features_df = psm_matched_features_df.withColumn(
            "outcome_total_campaign_revenue",
            F.col("baseline_60d_revenue").cast(DoubleType()),
        )

        result = CustomCode(self.custom_packages_path).custom_func(self.spark, psm_matched_features_df)
        self.data_handler.write(result)
