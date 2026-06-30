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
        """
        1.  Read the datasets: Use the read function of DataHandler class
            '''
                df = self.data_handler.read("<dataset_macro>")
            '''


        2.  Read runtime params:
            i.  Use the run_params function of DataHandler class
                '''
                    run_params = self.data_handler.run_params()
                '''
            ii. Get value and datatype of a parameter using below functions of DataHandler class
                '''
                    param_value = self.data_handler.get_run_parameter_value(param_name)
                    param_datatype = self.data_handler.get_run_parameter_datatype(param_name)
                '''

        3.  Ensure that your code is callable from this function
            '''
                CustomCode(self.custom_packages_path).custom_func(df)
            '''
           or write your code in this function
            '''
                macro = self.data_handler.get_column_for_macro('df1', 'ID')
                result = df1.join(df2, macro)
            '''

        4.  Write your extra outputs to file and persist them
            '''
                self.data_handler.save_output(extra_output_path)
            '''

        5.  MANDATORY: Write your code resultant dataframe
            '''
                self.data_handler.write(result)
            '''
        """

        psm_matched_features_df = self.data_handler.read("StratifiedFeatures")

        # Placebo: replace campaign-period outcome with pre-campaign 60-day revenue.
        # Mirrors ML-ATE-PLACEBO but runs on the exact-stratum matched cohort.
        # A significant τ̂ here means FeatureEnggStratify left pre-existing spend
        # imbalance despite exact blocking on baseline_12m_revenue_sum_bin.
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType
        psm_matched_features_df = psm_matched_features_df.withColumn(
            "outcome_campaign_product_revenue",
            F.col("baseline_60d_revenue").cast(DoubleType()),
        )

        result = CustomCode(self.custom_packages_path).custom_func(self.spark, psm_matched_features_df)
        self.data_handler.write(result)

        ##### START OF SAMPLE CODE ######
        # # Read datasets
        # owner_df = self.data_handler.read("owner")
        # partner_df = self.data_handler.read("partner")
        # partner_df2 = self.data_handler.read("partner2")

        # transcoded_result = self.transcode(owner_df, partner_df, partner_df2)
        # transcoded_result.show()

        # # Sample to make your code callable from this function
        # result = CustomCode(self.custom_packages_path).custom_func(owner_df, partner_df)


        # # Read RuntimeParams
        # run_params = self.data_handler.run_params()
        # param_value = self.data_handler.get_run_parameter_value("START_DATE")
        # param_datatype = self.data_handler.get_run_parameter_datatype("START_DATE")

        # # Only allowed when has_extra_outputs is set to true
        # result_file_path = f"{os.environ['PWD']}/result.csv"
        # result.toPandas().to_csv(result_file_path, index=False)
        # self.data_handler.save_output(result_file_path)

        # owner_data_file_path = f"{os.environ['PWD']}/owner.json"
        # owner_df.toPandas().to_json(owner_data_file_path, orient="records")
        # self.data_handler.save_output(owner_data_file_path)

        # partner_data_file_path = f"{os.environ['PWD']}/partner.json"
        # partner_df.toPandas().to_json(partner_data_file_path, orient="records")
        # self.data_handler.save_output(partner_data_file_path)

        # # or
        # # Sample code to write your transformations in this function
        # # result = owner_df.join(partner_df, 'ID', "inner")

        # # [MANDATORY STEP] Write the resultant dataset.
        # self.data_handler.write(result)

        ##### END OF SAMPLE CODE ######

    def transcode(self, owner_df: DataFrame, partner_df: DataFrame) -> DataFrame:
        base_py = BaseTablePy(
            df=owner_df,  # PySpark DataFrame
            table_name="owner",
            column_name="tp_id"
        )

        join_py1 = JoinTablePy(
            df=partner_df,  # PySpark DataFrame
            table_name="partner",
            column_name="rampid_meta",
            join_type="inner"
        )

        # join_py2 = JoinTablePy(
        #     df=partner_df2,  # PySpark DataFrame
        #     table_name="partner2",
        #     column_name="rampid",
        #     join_type="left"
        # )

        # Call the method
        return self.data_handler.transcode_rampid_join(base_py, [join_py1])

