import os
import shutil

from pyspark.sql import SparkSession, DataFrame


class BaseTablePy:
    def __init__(self, df: DataFrame, table_name: str, column_name: str):
        self.df = df
        self.table_name = table_name
        self.column_name = column_name

class JoinTablePy:
    def __init__(self, df: DataFrame, table_name: str, column_name: str, join_type: str):
        self.df = df
        self.table_name = table_name
        self.column_name = column_name
        self.join_type = join_type

class DataHandler:

    _OUTPUT_SPARK_TABLE = "result"

    def __init__(self, spark: SparkSession, has_extra_output=False, custom_packages_path='', macro_to_columns=None, runner={}, extra_output_macro_data_location_map={}, run_parameters={}):
        self.run_parameters = run_parameters
        self.spark = spark
        self.has_extra_output = has_extra_output
        self.custom_packages_path = custom_packages_path
        self.macro_to_columns = macro_to_columns
        self.runner = runner
        self.extra_output_macro_data_location_map = extra_output_macro_data_location_map


    def read(self, macro: str) -> DataFrame:
        """
        :param macro: The dataset macro from question builder UI
        :return: Spark dataframe for the requested dataset
        """
        return self.spark.sql(f"SELECT * FROM {macro}")

    def write(self, result: DataFrame):
        """
        :param result: Your resultant dataframe
        :return:
        """
        result.createOrReplaceTempView(self._OUTPUT_SPARK_TABLE)

    def save_output(self, file_path):
        if not self.has_extra_output:
            raise RuntimeError("Writing extra output is not allowed in this question")
        if os.path.isdir(file_path):
            shutil.copytree(file_path, f"{os.environ['OUTPUT_EXTRA_PATH']}/{file_path.split('/')[-1]}")
        elif os.path.isfile(file_path):
            shutil.copy(file_path, f"{os.environ['OUTPUT_EXTRA_PATH']}/{file_path.split('/')[-1]}")

    def read_output(self, macro, file_name) -> DataFrame:
        if not self.has_extra_output:
            raise RuntimeError("Extra output is not allowed in this question")

        extra_output_base_path = self.extra_output_macro_data_location_map.get(macro, '')
        if extra_output_base_path == '':
            raise RuntimeError(f"Extra output location not found for macro {macro}")

        extra_output_file_path = f"{extra_output_base_path}/{file_name}"

        if file_name.endswith(".parquet"):
            return self.spark.read.parquet(extra_output_file_path)
        elif file_name.endswith(".csv"):
            return self.spark.read.option("header", "true").csv(extra_output_file_path)
        elif file_name.endswith(".json"):
            return self.spark.read.json(extra_output_file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_name}")

    def get_run_parameter_value(self, parameter_name):
        if parameter_name in self.run_parameters:
            return self.run_parameters[parameter_name][0]
        raise RuntimeError(f"Run parameter not found: {parameter_name}")

    def get_run_parameter_datatype(self, parameter_name):
        if parameter_name in self.run_parameters:
            return self.run_parameters[parameter_name][1]
        raise RuntimeError(f"Run parameter not found: {parameter_name}")

    def run_params(self):
        return self.run_parameters

    def get_column_for_macro(self, table_macro: str, column_macro: str):
        """
        :param table_macro: table name macro
        :param column_macro: column name macro
        :return: the actual column name if it exists, otherwise return column_macro.
        """
        return self.macro_to_columns.get(table_macro, {}).get(column_macro, column_macro)

    def transcode_rampid_join(self, base_table: BaseTablePy, join_tables: list[JoinTablePy]) -> DataFrame:
        """
        Calls the Scala transcodeRampIdJoin method with Python-friendly BaseTable and JoinTable objects.

        Parameters
        ----------
        base_table : BaseTablePy
            The base table containing (df: DataFrame, tableName: String, columnName: String).
        join_tables : list[JoinTablePy]
            A list of join tables, each with (df: DataFrame, tableName: String, columnName: String, joinType: String).

        Returns
        -------
        PySparkDataFrame
            The joined DataFrame after transcoding logic from Scala.
        """

        jvm = self.spark.sparkContext._jvm

        # Get Scala case class references
        BaseTableScala = jvm.com.habu.cleancompute.transcode.BaseTable
        JoinTableScala = jvm.com.habu.cleancompute.transcode.JoinTable

        # Create Scala BaseTable instance
        scala_base = BaseTableScala(
            base_table.df._jdf,
            base_table.table_name,
            base_table.column_name
        )

        # Convert Python JoinTablePy list to Scala Seq[JoinTable]
        scala_joins = jvm.scala.collection.JavaConverters.asScalaBufferConverter([
            JoinTableScala(jt.df._jdf, jt.table_name, jt.column_name, jt.join_type)
            for jt in join_tables
        ]).asScala()

        # Call Scala method
        result_jdf = self.runner.transcodeRampIdJoin(scala_base, scala_joins)

        # Convert back to PySpark DataFrame
        return DataFrame(result_jdf, self.spark)
