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

        result = CustomCode(self.custom_packages_path).custom_func(self.spark, psm_matched_features_df)
        self.data_handler.write(result)
