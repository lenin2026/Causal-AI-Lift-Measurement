import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from functools import reduce

# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from dowhy import CausalModel
# import pandas as pd


class CustomCode:

    def __init__(self, custom_packages_path=''):
        # The root directory where the custom artifact is installed.
        self.custom_packages_path = custom_packages_path

    def custom_func(self, spark: SparkSession, final_df: DataFrame) -> DataFrame:

        # ── Map FeatureEnggStratify (q84B) columns to ML-ATE naming ─────────
        # outcome_total_campaign_revenue is added by FeatureEngg v1.3+; fall back to
        # outcome_campaign_product_revenue when running against older AllFeatures data.
        _outcome_src = (
            "outcome_total_campaign_revenue"
            if "outcome_total_campaign_revenue" in final_df.columns
            else "outcome_campaign_product_revenue"
        )
        final_df = (
            final_df
            .withColumn("post_campaign_total_order_value",
                        F.col(_outcome_src).cast(DoubleType()))
            .withColumn("pre_campaign_total_order_value",
                        F.col("baseline_12m_revenue_sum").cast(DoubleType()))
            .withColumn("baseline_60d_revenue",
                        F.col("baseline_60d_revenue").cast(DoubleType()))
        )

        # Majority gender per addressLink (used by StringIndexer)
        final_df = final_df.withColumn(
            "gender",
            F.when(F.col("num_male_in_addresslink") > F.col("num_female_in_addresslink"), F.lit("male"))
            .when(F.col("num_female_in_addresslink") > F.col("num_male_in_addresslink"), F.lit("female"))
            .otherwise(F.lit("unknown")),
        )

        # state_label carries the state value from q84B
        final_df = final_df.withColumn("state", F.col("state_label"))

        # Weighted midpoint age from household age-bucket counts
        _age_buckets = [
            ("age_18_24_count", 21.0),
            ("age_25_34_count", 29.5),
            ("age_35_44_count", 39.5),
            ("age_45_54_count", 49.5),
            ("age_55_64_count", 59.5),
            ("age_65_74_count", 69.5),
            ("age_75_84_count", 79.5),
            ("age_85_plus_count", 90.0),
        ]
        _age_total    = reduce(lambda a, b: a + b, [F.col(c) for c, _ in _age_buckets])
        _age_weighted = reduce(lambda a, b: a + b, [F.lit(m) * F.col(c) for c, m in _age_buckets])
        final_df = final_df.withColumn(
            "est_age",
            F.when(_age_total > 0, _age_weighted / _age_total).otherwise(F.lit(40.0)).cast(DoubleType()),
        )

        # Weighted average income code (1–35) from household income-code counts.
        # Higher code generally indicates higher income in LiveRamp IXI; codes outside 1–35 are excluded.
        _income_total    = reduce(lambda a, b: a + b,
                                  [F.col(f"num_hh_income_code_{i}_in_addresslink") for i in range(1, 36)])
        _income_weighted = reduce(lambda a, b: a + b,
                                  [F.lit(float(i)) * F.col(f"num_hh_income_code_{i}_in_addresslink")
                                   for i in range(1, 36)])
        final_df = final_df.withColumn(
            "est_income_code",
            F.when(_income_total > 0, _income_weighted / _income_total).otherwise(F.lit(18.0)).cast(DoubleType()),
        )

        # -------------------------- CAUSAL MODELING --------------------------

        post_spend_99p, pre_spend_99p, b60d_99p = final_df.approxQuantile(
            ["post_campaign_total_order_value", "pre_campaign_total_order_value", "baseline_60d_revenue"], [0.99], 0.01
        )
        post_spend_99p = post_spend_99p[0]
        pre_spend_99p  = pre_spend_99p[0]
        b60d_99p       = b60d_99p[0]

        final_df = final_df.withColumn(
            "post_campaign_total_order_value",
            F.when(F.col("post_campaign_total_order_value") > post_spend_99p, F.lit(post_spend_99p))
            .otherwise(F.col("post_campaign_total_order_value")),
        )
        final_df = final_df.withColumn(
            "pre_campaign_total_order_value",
            F.when(F.col("pre_campaign_total_order_value") > pre_spend_99p, F.lit(pre_spend_99p))
            .otherwise(F.col("pre_campaign_total_order_value")),
        )
        final_df = final_df.withColumn(
            "baseline_60d_revenue",
            F.when(F.col("baseline_60d_revenue") > b60d_99p, F.lit(b60d_99p))
            .otherwise(F.col("baseline_60d_revenue")),
        )

        # ── Encode categorical features ────────────────────────────────────
        for col, out in [("gender", "gender_index"),
                         ("state",  "state_index")]:
            final_df = StringIndexer(inputCol=col, outputCol=out, handleInvalid="keep").fit(final_df).transform(final_df)

        # Feature Assembly
        ml_data = final_df
        assembler = VectorAssembler(
            inputCols=[
                "pre_campaign_total_order_value",
                "baseline_60d_revenue",
                "baseline_purchase_tenure_days",
                "days_since_last_baseline_purchase",
                "baseline_12m_avg_order_value",
                "baseline_12m_avg_items_per_order",
                "baseline_12m_orders",
                "est_age",
                "est_income_code",
                "gender_index",
                "state_index",
            ],
            outputCol="features",
            handleInvalid="skip",
        )
        ml_data = assembler.transform(ml_data)

        # ── Cross-Fitting (K-Fold Cross Validation) ────────────────────────
        num_folds = 5
        # Assign random folds
        ml_data = ml_data.withColumn("fold", (F.rand(seed=42) * num_folds).cast(IntegerType()))

        lr_y = LinearRegression(labelCol="post_campaign_total_order_value", featuresCol="features",
                                predictionCol="y_hat", maxIter=10, tol=1e-3, loss="huber", epsilon=1.35)
        lr_t = LogisticRegression(featuresCol="features", labelCol="treatment",
                                  probabilityCol="probability", predictionCol="t_pred",
                                  maxIter=10, regParam=0.1, tol=1e-3)

        oof_predictions = []

        # Iterate through folds to generate unbiased out-of-fold predictions
        for k in range(num_folds):
            train_df = ml_data.filter(F.col("fold") != k)
            test_df  = ml_data.filter(F.col("fold") == k)

            model_y_k = lr_y.fit(train_df)
            model_t_k = lr_t.fit(train_df)

            scored_k = model_y_k.transform(test_df)
            scored_k = model_t_k.transform(scored_k)

            oof_predictions.append(scored_k)

        # Union all out-of-fold predictions back together
        ml_data_cv = reduce(DataFrame.unionByName, oof_predictions)

        # Calculate unbiased residuals
        ml_data_cv = ml_data_cv.withColumn("y_res", F.col("post_campaign_total_order_value") - F.col("y_hat"))
        ml_data_cv = ml_data_cv.withColumn("t_hat", vector_to_array(F.col("probability"))[1])
        ml_data_cv = ml_data_cv.withColumn(
            "t_hat",
            F.when(F.col("t_hat") > 0.95, 0.95).when(F.col("t_hat") < 0.05, 0.05).otherwise(F.col("t_hat")),
        )
        ml_data_cv = ml_data_cv.withColumn("t_res", F.col("treatment") - F.col("t_hat"))

        # ATE with residuals (Causal Link b/w Y_res and T_res)
        final_assembler = VectorAssembler(inputCols=["t_res"], outputCol="final_features", handleInvalid="skip")
        final_ml_data   = final_assembler.transform(ml_data_cv)

        causal_model    = LinearRegression(featuresCol="final_features", labelCol="y_res",
                                           predictionCol="final_pred", fitIntercept=True,
                                           maxIter=10, tol=1e-3, solver="normal")
        final_estimate  = causal_model.fit(final_ml_data)

        # Evaluation ----------------------------------------------------------

        # Outcome R-Squared
        eval_y = RegressionEvaluator(labelCol="post_campaign_total_order_value",
                                     predictionCol="y_hat", metricName="r2")
        r2_o = float(eval_y.evaluate(ml_data_cv))

        # Treatment AUC
        eval_t = BinaryClassificationEvaluator(labelCol="treatment",
                                               rawPredictionCol="rawPrediction",
                                               metricName="areaUnderROC")
        auc_t = float(eval_t.evaluate(ml_data_cv))

        # Incremental Lift ----------------------------------------------------
        causal_summary = final_estimate.summary
        lift_value     = float(final_estimate.coefficients[0])
        lift_std_err   = float(causal_summary.coefficientStandardErrors[0])
        lift_p_value   = float(causal_summary.pValues[0])
        lift_ci_lower  = lift_value - 1.96 * lift_std_err
        lift_ci_upper  = lift_value + 1.96 * lift_std_err

        stats = ml_data_cv.agg(
            F.count("*").alias("total_count"),
            F.sum(F.col("treatment").cast(DoubleType())).alias("treated_count"),
            # ── Why unconditional average (no Y > 0 filter) ───────────────────
            # lift_value (τ) is the Double-ML ATE: the average incremental spend
            # caused by ad exposure across the ENTIRE treated population, including
            # households that spent $0 during the campaign window. Formally:
            #   τ = E[Y(1)] − E[Y(0)]   (unconditional, all T=1 units)
            #
            # avg_treatment_amount is used to back-calculate the counterfactual
            # baseline (expected_amount), the spend we would have observed in the
            # treatment group had they NOT been exposed:
            #   expected_amount = E[Y|T=1] − τ  ≈  E[Y(0)|T=1]
            #
            # If we filter to Y > 0 we compute E[Y|T=1, Y>0] instead of E[Y|T=1].
            # For a typical retail campaign with a 5–15% conversion rate,
            # E[Y|T=1, Y>0] is 7–20× larger than E[Y|T=1] (only converters vs.
            # all exposed). Subtracting an unconditional τ from a conditional
            # average breaks the math:
            #   WRONG:  expected_amount = E[Y|T=1, Y>0] − τ   ← inflated denominator
            #   CORRECT: expected_amount = E[Y|T=1]      − τ  ← matched estimands
            #
            # The inflated denominator suppresses lift_percent and its confidence
            # intervals (lift_pct_ci_lower, lift_pct_ci_upper), making the reported
            # percentage lift appear far smaller than the true effect.
            F.avg(F.when(
                F.col("treatment") == 1,
                F.col("post_campaign_total_order_value"),
            )).alias("avg_treatment_spend"),
        ).collect()[0]
        total_count         = stats["total_count"]
        treated_count       = stats["treated_count"]
        avg_treatment_spend = stats["avg_treatment_spend"] or 0.0

        expected_spend = avg_treatment_spend - lift_value
        lift_percent   = (lift_value / expected_spend * 100) if expected_spend != 0 else 0

        lift_pct_ci_lower = (lift_value - 1.96 * lift_std_err) / expected_spend * 100 if expected_spend != 0 else 0
        lift_pct_ci_upper = (lift_value + 1.96 * lift_std_err) / expected_spend * 100 if expected_spend != 0 else 0

        # ── CATE: effect modifiers ─────────────────────────────────────────
        # has_baseline_purchase → buyers vs. non-buyers respond very differently
        # days_since_last_baseline_purchase → recently active vs. lapsed (>= 30 days = lapsed)
        # est_age → young (<= 35) vs. senior (>= 55) segments
        # est_income_code >= 20 → above-median income (codes 1–35, higher = higher income)

        # PSMMatchedFeatures produced by ML-PSM wheels prior to the explicit
        # output_cols refactor may not carry has_baseline_purchase. Derive it
        # from days_since_last_baseline_purchase (always present — it is an
        # assembler input, so a missing value would have been caught above).
        # FeatureEngg sets days_since_last_baseline_purchase = 366 for non-buyers,
        # so the inverse is: any value < 366 means at least one baseline purchase.
        if "has_baseline_purchase" not in ml_data_cv.columns:
            ml_data_cv = ml_data_cv.withColumn(
                "has_baseline_purchase",
                F.when(F.col("days_since_last_baseline_purchase") < 366, F.lit(1))
                 .otherwise(F.lit(0))
                 .cast(IntegerType()),
            )

        ml_data_cate = (
            ml_data_cv
            .withColumn("seg_buyer",       F.col("has_baseline_purchase").cast(DoubleType()))
            .withColumn("seg_lapsed",      F.when(F.col("days_since_last_baseline_purchase") >= 30,
                                                  F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn("seg_young",       F.when(F.col("est_age") <= 35, F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn("seg_senior",      F.when(F.col("est_age") >= 55, F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn("seg_high_income", F.when(F.col("est_income_code") >= 20,
                                                  F.lit(1.0)).otherwise(F.lit(0.0)))
        )

        # Interaction terms: modifier × t_res
        ml_data_cate = (
            ml_data_cate
            .withColumn("t_res_x_buyer",       F.col("t_res") * F.col("seg_buyer"))
            .withColumn("t_res_x_lapsed",      F.col("t_res") * F.col("seg_lapsed"))
            .withColumn("t_res_x_young",       F.col("t_res") * F.col("seg_young"))
            .withColumn("t_res_x_senior",      F.col("t_res") * F.col("seg_senior"))
            .withColumn("t_res_x_high_income", F.col("t_res") * F.col("seg_high_income"))
        )

        cate_feature_cols = ["t_res", "t_res_x_buyer", "t_res_x_lapsed", "t_res_x_young",
                             "t_res_x_senior", "t_res_x_high_income"]
        cate_assembler = VectorAssembler(inputCols=cate_feature_cols, outputCol="cate_features",
                                         handleInvalid="skip")
        cate_ml_data   = cate_assembler.transform(ml_data_cate)
        cate_model     = LinearRegression(featuresCol="cate_features", labelCol="y_res",
                                          predictionCol="cate_prediction", fitIntercept=True)
        cate_estimate  = cate_model.fit(cate_ml_data)

        # Coefficients: [t_res, t_res_x_buyer, t_res_x_lapsed, t_res_x_young, t_res_x_senior, t_res_x_high_income]
        coefs         = cate_estimate.coefficients
        cate_summary  = cate_estimate.summary
        ate_base             = float(coefs[0])
        cate_buyer_delta     = float(coefs[1])
        cate_lapsed_delta    = float(coefs[2])
        cate_young_delta     = float(coefs[3])
        cate_senior_delta    = float(coefs[4])
        cate_high_income_delta = float(coefs[5])

        # Derived CATE per segment (additive on top of base ATE)
        cate_buyer       = ate_base + cate_buyer_delta
        cate_lapsed      = ate_base + cate_lapsed_delta
        cate_young       = ate_base + cate_young_delta
        cate_senior      = ate_base + cate_senior_delta
        cate_high_income = ate_base + cate_high_income_delta

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
