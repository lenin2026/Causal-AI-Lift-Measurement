import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator

# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from dowhy import CausalModel
# import pandas as pd


class CustomCode:

    def __init__(self, custom_packages_path=''):
        # The root directory where the custom artifact is installed.
        self.custom_packages_path = custom_packages_path

    def custom_func(self, spark: SparkSession, final_df: DataFrame) -> DataFrame:

        # -------------------------- CAUSAL MODELING --------------------------

        post_spend_99p, pre_spend_99p = final_df.approxQuantile(["post_campaign_total_order_value", "pre_campaign_total_order_value"], [0.99], 0.01)
        post_spend_99p = post_spend_99p[0]
        pre_spend_99p = pre_spend_99p[0]

        final_df = final_df.withColumn("post_campaign_total_order_value", F.when(F.col("post_campaign_total_order_value") > post_spend_99p, \
                                                                                 F.lit(post_spend_99p))\
                                                                           .otherwise(F.col("post_campaign_total_order_value")))
        final_df = final_df.withColumn("pre_campaign_total_order_value", F.when(F.col("pre_campaign_total_order_value") > pre_spend_99p, \
                                                                                F.lit(pre_spend_99p))\
                                                                          .otherwise(F.col("pre_campaign_total_order_value")))

        # ── Encode categorical features ────────────────────────────────────
        for col, out in [("gender",           "gender_index"),
                         ("state",            "state_index"),
                         ("device_platform",  "device_platform_index"),
                         ("impression_device","impression_device_index"),
                         ("placement_type",   "placement_type_index")]:
            final_df = StringIndexer(inputCol=col, outputCol=out, handleInvalid="keep").fit(final_df).transform(final_df)

        # Feature Assembly
        ml_data = final_df
        assembler = VectorAssembler(inputCols=["pre_campaign_tenure_days", "pre_campaign_conversion_recency", \
                                               "pre_campaign_avg_order_value", "pre_campaign_avg_items_per_order", \
                                            #    "exposure_frequency", "days_into_campaign_at_exposure", \
                                               "pre_campaign_conversion_count", "age", "hh_income", "gender_index", "state_index", \
                                            #    "device_platform_index", "impression_device_index", "placement_type_index"
                                               ], outputCol="features", handleInvalid="skip")
        ml_data = assembler.transform(ml_data)

        # Outcome model
        lr_y = LinearRegression(labelCol="post_campaign_total_order_value", featuresCol="features", predictionCol="y_hat", maxIter=10, tol=1e-3)
        model_y = lr_y.fit(ml_data)

        # Unexplained spend
        ml_data = model_y.transform(ml_data).withColumn("y_res", F.col("post_campaign_total_order_value") - F.col("y_hat"))

        # Treatment model
        lr_t = LogisticRegression(featuresCol="features", labelCol="treatment", probabilityCol="probability", predictionCol="t_pred", maxIter=10, regParam=0.1, tol=1e-3)
        # lr_t = LinearRegression(featuresCol="features", labelCol="treatment", predictionCol="t_hat", maxIter=10, tol=1e-3)
        model_t = lr_t.fit(ml_data)
        
        # Unexplained exposure
        ml_data = model_t.transform(ml_data)
        ml_data = ml_data.withColumn("t_hat", vector_to_array(F.col("probability"))[1])\
                         .withColumn("t_res", F.col("treatment") - F.col("t_hat"))

        # ATE with residuals (Causal Link b/w Y_res and T_res)
        # How much of y_res can be explained by t_res?
        final_assembler = VectorAssembler(inputCols=["t_res"], outputCol="final_features", handleInvalid="skip")
        final_ml_data = final_assembler.transform(ml_data)

        causal_model = LinearRegression(featuresCol="final_features", labelCol="y_res", predictionCol="final_pred", fitIntercept=True, maxIter=10, tol=1e-3, solver="normal")
        final_estimate = causal_model.fit(final_ml_data)
        # # ml_data.unpersist()

        # Evaluation ----------------------------------------------------------

        # Outcome R-Squared 
        eval_y = RegressionEvaluator(labelCol="post_campaign_total_order_value", predictionCol="y_hat", metricName="r2")
        r2_o = float(eval_y.evaluate(ml_data))

        # Treatment AUC
        eval_t = BinaryClassificationEvaluator(labelCol="treatment", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc_t = float(eval_t.evaluate(ml_data))

        # Incremental Lift ----------------------------------------------------
        # Coefficient of t_res: For 1 unit of t_res, how much y_res change did we see?
        # Evaluation: CI and p-value from causal model
        # Spark LinearRegression summary provides standard errors directly
        causal_summary  = final_estimate.summary
        lift_value      = float(final_estimate.coefficients[0])
        lift_std_err    = float(causal_summary.coefficientStandardErrors[0])
        lift_t_stat     = float(causal_summary.tValues[0])
        lift_p_value    = float(causal_summary.pValues[0])
        # 95% confidence interval: estimate ± 1.96 * std_error
        lift_ci_lower   = lift_value - 1.96 * lift_std_err
        lift_ci_upper   = lift_value + 1.96 * lift_std_err

        stats = ml_data.agg(F.count("*").alias("total_count"),
                            F.sum(F.col("treatment").cast(DoubleType())).alias("treated_count"),
                            # avg_treatment_spend only for Q1 — exposed AND converted
                            F.avg(F.when((F.col("treatment") == 1) & (F.col("post_campaign_total_order_value") > 0), \
                                          F.col("post_campaign_total_order_value"))).alias("avg_treatment_spend"),
                            ).collect()[0]
        total_count = stats["total_count"]
        treated_count = stats["treated_count"]
        avg_treatment_spend = stats["avg_treatment_spend"] or 0.0
        
        expected_spend = avg_treatment_spend - lift_value
        lift_percent = (lift_value / expected_spend * 100) if expected_spend != 0 else 0

        lift_pct_ci_lower  = (lift_value - 1.96 * lift_std_err) / expected_spend * 100 if expected_spend != 0 else 0
        lift_pct_ci_upper  = (lift_value + 1.96 * lift_std_err) / expected_spend * 100 if expected_spend != 0 else 0

        # ── CATE: effect modifiers ─────────────────────────────────────────
        # Which variables plausibly moderate the ad effect on spend?
        
        # pre_campaign_has_conversion  → buyers vs. non-buyers respond very differently
        # pre_campaign_tenure_days     → new vs. tenured customers
        # pre_campaign_avg_order_value → low/mid/high value segments
        # pre_campaign_conversion_recency → recently active vs. lapsed
        
        # Method: add interaction terms (modifier × t_res) to the causal regression.
        # The coefficient on each interaction IS the CATE slope for that modifier.

        # Bucket continuous modifiers into segments first (keeps interpretation clean)
        ml_data_cate = ml_data.withColumn("seg_buyer", F.col("pre_campaign_has_conversion").cast(DoubleType()))\
                              .withColumn("seg_lapsed", F.when(F.col("pre_campaign_conversion_recency") >= 30, F.lit(1.0)).otherwise(F.lit(0.0)))\
                              .withColumn("seg_young", F.when(F.col("age") <= 35, F.lit(1.0)).otherwise(F.lit(0.0)))\
                              .withColumn("seg_senior", F.when(F.col("age") >= 55, F.lit(1.0)).otherwise(F.lit(0.0)))\
                              .withColumn("seg_high_income", F.when(F.col("hh_income") >= 75000, F.lit(1.0)).otherwise(F.lit(0.0)))                     

        # Interaction terms: modifier × t_res
        ml_data_cate = ml_data_cate.withColumn("t_res_x_buyer", F.col("t_res") * F.col("seg_buyer")) \
                                   .withColumn("t_res_x_lapsed", F.col("t_res") * F.col("seg_lapsed")) \
                                   .withColumn("t_res_x_young", F.col("t_res") * F.col("seg_young")) \
                                   .withColumn("t_res_x_senior", F.col("t_res") * F.col("seg_senior")) \
                                   .withColumn("t_res_x_high_income", F.col("t_res") * F.col("seg_high_income"))

        cate_feature_cols = ["t_res", "t_res_x_buyer", "t_res_x_lapsed", "t_res_x_young", "t_res_x_senior", "t_res_x_high_income"]
        cate_assembler  = VectorAssembler(inputCols=cate_feature_cols, outputCol="cate_features", handleInvalid="skip")
        cate_ml_data    = cate_assembler.transform(ml_data_cate)
        cate_model      = LinearRegression(featuresCol="cate_features", labelCol="y_res",
                                           predictionCol="cate_prediction", fitIntercept=True)
        cate_estimate   = cate_model.fit(cate_ml_data)

        # Coefficients: [t_res, t_res_x_buyer, t_res_x_new, t_res_x_high_aov, t_res_x_lapsed]
        coefs = cate_estimate.coefficients
        cate_summary = cate_estimate.summary
        ate_base          = float(coefs[0])   # baseline ATE (non-buyer, tenured, low-AOV, active)
        cate_buyer_delta  = float(coefs[1])   # additional lift for existing buyers
        cate_lapsed_delta = float(coefs[2])   # additional lift for lapsed customers
        cate_young_delta  = float(coefs[3])   # additional lift for young customers
        cate_senior_delta = float(coefs[4])   # additional lift for senior customers
        cate_high_income_delta = float(coefs[5])   # additional lift for high income customers

        # Derived CATE per segment (additive on top of base ATE)
        cate_buyer    = ate_base + cate_buyer_delta
        cate_lapsed   = ate_base + cate_lapsed_delta
        cate_young    = ate_base + cate_young_delta
        cate_senior   = ate_base + cate_senior_delta
        cate_high_income = ate_base + cate_high_income_delta

        # cate_buyer_ci_lower = cate_buyer - 1.96 * cate_summary.coefficientStandardErrors[1]
        # cate_buyer_ci_upper = cate_buyer + 1.96 * cate_summary.coefficientStandardErrors[1]
        # cate_lapsed_ci_lower = cate_lapsed - 1.96 * cate_summary.coefficientStandardErrors[2]
        # cate_lapsed_ci_upper = cate_lapsed + 1.96 * cate_summary.coefficientStandardErrors[2]
        # cate_young_ci_lower = cate_young - 1.96 * cate_summary.coefficientStandardErrors[3]
        # cate_young_ci_upper = cate_young + 1.96 * cate_summary.coefficientStandardErrors[3]
        # cate_senior_ci_lower = cate_senior - 1.96 * cate_summary.coefficientStandardErrors[4]
        # cate_senior_ci_upper = cate_senior + 1.96 * cate_summary.coefficientStandardErrors[4]
        # cate_high_income_ci_lower = cate_high_income - 1.96 * cate_summary.coefficientStandardErrors[5]
        # cate_high_income_ci_upper = cate_high_income + 1.96 * cate_summary.coefficientStandardErrors[5]

        try:
            cate_buyer_p_value       = float(cate_summary.pValues[1])
            cate_lapsed_p_value      = float(cate_summary.pValues[2])
            cate_young_p_value       = float(cate_summary.pValues[3])
            cate_senior_p_value      = float(cate_summary.pValues[4])
            cate_high_income_p_value = float(cate_summary.pValues[5])
        except:
            cate_buyer_p_value       = -1.0
            cate_lapsed_p_value      = -1.0
            cate_young_p_value       = -1.0
            cate_senior_p_value      = -1.0
            cate_high_income_p_value = -1.0

        result = spark.createDataFrame(
            [(
                lift_value, lift_percent, avg_treatment_spend, expected_spend,
                total_count, treated_count, 
                r2_o, auc_t,
                lift_pct_ci_lower, lift_pct_ci_upper,
                lift_ci_lower, lift_ci_upper, lift_p_value,
                ate_base, cate_buyer, cate_lapsed, cate_young, cate_senior, cate_high_income,
                cate_buyer_p_value, cate_lapsed_p_value, cate_young_p_value, cate_senior_p_value, cate_high_income_p_value
            )],
            [
                "incremental_lift", "lift_percent", "avg_treatment_amount", "expected_amount",
                "total_row_count", "treated_count", 
                "outcome_r2", "treatment_auc", 
                "lift_pct_ci_lower", "lift_pct_ci_upper",
                "lift_ci_lower", "lift_ci_upper", "lift_p_value",
                "ate_base", "cate_existing_buyer", "cate_lapsed", "cate_young", "cate_senior", "cate_high_income",
                "cate_existing_buyer_p_value", "cate_lapsed_p_value", "cate_young_p_value", "cate_senior_p_value", "cate_high_income_p_value"
            ],
        )
        
        return result
