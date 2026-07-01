import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from functools import reduce


class CustomCode:

    def __init__(self, custom_packages_path=''):
        self.custom_packages_path = custom_packages_path

    def custom_func(self, spark: SparkSession, final_df: DataFrame) -> DataFrame:

        # ── Column mapping ────────────────────────────────────────────────
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
        )

        # ── Ordinal encoding: revenue bin (zero=0 … 12000_plus=12) ───────
        # StringIndexer would assign frequency-based indices, destroying the
        # natural dollar-value order. Manual mapping preserves it for GBT splits.
        final_df = final_df.withColumn(
            "revenue_bin_ordinal",
            F.when(F.col("baseline_12m_revenue_sum_bin") == "zero",         F.lit(0.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "lt_10",         F.lit(1.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "10_to_49",      F.lit(2.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "50_to_99",      F.lit(3.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "100_to_249",    F.lit(4.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "250_to_499",    F.lit(5.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "500_to_999",    F.lit(6.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "1000_to_1999",  F.lit(7.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "2000_to_3499",  F.lit(8.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "3500_to_4999",  F.lit(9.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "5000_to_7499",  F.lit(10.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "7500_to_11999", F.lit(11.0))
            .when(F.col("baseline_12m_revenue_sum_bin") == "12000_plus",    F.lit(12.0))
            .otherwise(F.lit(0.0))
            .cast(DoubleType())
        )

        # ── Ordinal encoding: buyer recency (no_12m=0, lapsed=1, recent=2) ─
        final_df = final_df.withColumn(
            "buyer_label_ordinal",
            F.when(F.col("baseline_buyer_label") == "recent_buyer",  F.lit(2.0))
            .when(F.col("baseline_buyer_label") == "lapsed_buyer",   F.lit(1.0))
            .otherwise(F.lit(0.0))
            .cast(DoubleType())
        )

        # ── Derive majority gender from household counts ───────────────────
        final_df = final_df.withColumn(
            "gender",
            F.when(F.col("num_male_in_addresslink") > F.col("num_female_in_addresslink"), F.lit("male"))
            .when(F.col("num_female_in_addresslink") > F.col("num_male_in_addresslink"), F.lit("female"))
            .otherwise(F.lit("unknown")),
        )

        # ── Label-encode nominal categoricals ─────────────────────────────
        # GBT splits on numeric thresholds; StringIndexer assigns frequency-based
        # indices which is acceptable for nominal features (no natural order).
        for col_name, out_name in [
            ("gender",                          "gender_index"),
            ("state_label",                     "state_index"),
            ("poc_label",                       "poc_index"),
            ("campaign_product_affinity_label", "affinity_index"),
        ]:
            final_df = StringIndexer(
                inputCol=col_name, outputCol=out_name, handleInvalid="keep"
            ).fit(final_df).transform(final_df)

        # ── has_baseline_purchase fallback ────────────────────────────────
        if "has_baseline_purchase" not in final_df.columns:
            final_df = final_df.withColumn(
                "has_baseline_purchase",
                F.when(F.col("days_since_last_baseline_purchase") < 366, F.lit(1))
                 .otherwise(F.lit(0))
                 .cast(IntegerType()),
            )

        # ── 99th-percentile winsorisation (outcome + key spend features) ──
        post_99p, pre_99p, b60d_99p, bcpr_99p = final_df.approxQuantile(
            ["post_campaign_total_order_value", "pre_campaign_total_order_value",
             "baseline_60d_revenue", "baseline_campaign_product_revenue"],
            [0.99], 0.01
        )
        post_99p  = post_99p[0]
        pre_99p   = pre_99p[0]
        b60d_99p  = b60d_99p[0]
        bcpr_99p  = bcpr_99p[0]

        final_df = (
            final_df
            .withColumn("post_campaign_total_order_value",
                        F.when(F.col("post_campaign_total_order_value") > post_99p, F.lit(post_99p))
                        .otherwise(F.col("post_campaign_total_order_value")))
            .withColumn("pre_campaign_total_order_value",
                        F.when(F.col("pre_campaign_total_order_value") > pre_99p, F.lit(pre_99p))
                        .otherwise(F.col("pre_campaign_total_order_value")))
            .withColumn("baseline_60d_revenue",
                        F.when(F.col("baseline_60d_revenue") > b60d_99p, F.lit(b60d_99p))
                        .otherwise(F.col("baseline_60d_revenue").cast(DoubleType())))
            .withColumn("baseline_campaign_product_revenue",
                        F.when(F.col("baseline_campaign_product_revenue") > bcpr_99p, F.lit(bcpr_99p))
                        .otherwise(F.col("baseline_campaign_product_revenue").cast(DoubleType())))
        )

        # ── Feature assembly ──────────────────────────────────────────────
        # Spend / frequency features
        continuous_cols = [
            "pre_campaign_total_order_value",
            "baseline_60d_revenue",
            "baseline_purchase_tenure_days",
            "days_since_last_baseline_purchase",
            "baseline_12m_avg_order_value",
            "baseline_12m_avg_items_per_order",
            "baseline_12m_orders",
            "baseline_12m_quantity_sum",
            "baseline_60d_orders",
            "baseline_campaign_product_revenue",
            "campaign_product_revenue_share",
            "recent_60d_revenue_share",
            "recent_60d_order_share",
        ]
        # Binary engagement / recency flags
        binary_cols = [
            "has_baseline_purchase",
            "prior_campaign_product_buyer",
            "recent_60d_buyer",
            "lapsed_60d_buyer",
        ]
        # Raw age-bucket counts — GBT exploits the full distribution rather than
        # collapsing it to a single weighted midpoint (est_age)
        age_cols = [
            "age_18_24_count", "age_25_34_count", "age_35_44_count", "age_45_54_count",
            "age_55_64_count", "age_65_74_count", "age_75_84_count", "age_85_plus_count",
        ]
        # Raw income-code counts (codes 1–35) — preserves income distribution shape;
        # est_income_code single average hides bimodal rich/poor households
        income_cols = [f"num_hh_income_code_{i}_in_addresslink" for i in range(1, 36)]
        # Ordinal-encoded categoricals (manual mapping preserves natural order)
        ordinal_cols = ["revenue_bin_ordinal", "buyer_label_ordinal"]
        # Frequency-indexed nominal categoricals
        indexed_cols = ["gender_index", "state_index", "poc_index", "affinity_index"]
        # Exposure intensity
        exposure_cols = ["exposure_frequency_deduped", "person_record_count"]

        all_feature_cols = (
            continuous_cols + binary_cols + age_cols + income_cols +
            ordinal_cols + indexed_cols + exposure_cols
        )

        assembler = VectorAssembler(
            inputCols=all_feature_cols,
            outputCol="features",
            handleInvalid="keep",  # GBT handles NaN/missing natively; no rows dropped
        )
        ml_data = assembler.transform(final_df)

        # ── Cross-fitting: K=5 out-of-fold residuals ──────────────────────
        num_folds = 5
        ml_data = ml_data.withColumn("fold", (F.rand(seed=42) * num_folds).cast(IntegerType()))

        # Outcome nuisance: GBTRegressor with absolute (L1) loss mirrors the
        # robustness of Huber regression used in the linear baseline.
        gbt_y = GBTRegressor(
            labelCol="post_campaign_total_order_value",
            featuresCol="features",
            predictionCol="y_hat",
            maxIter=50,
            maxDepth=5,
            stepSize=0.1,
            subsamplingRate=0.8,
            featureSubsetStrategy="auto",
            lossType="absolute",
            seed=42,
        )
        # Treatment nuisance: GBTClassifier for propensity score estimation
        gbt_t = GBTClassifier(
            featuresCol="features",
            labelCol="treatment",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            predictionCol="t_pred",
            maxIter=50,
            maxDepth=5,
            stepSize=0.1,
            subsamplingRate=0.8,
            featureSubsetStrategy="auto",
            seed=42,
        )

        oof_predictions = []
        for k in range(num_folds):
            train_df = ml_data.filter(F.col("fold") != k)
            test_df  = ml_data.filter(F.col("fold") == k)

            model_y_k = gbt_y.fit(train_df)
            model_t_k = gbt_t.fit(train_df)

            scored_k = model_y_k.transform(test_df)
            scored_k = model_t_k.transform(scored_k)

            oof_predictions.append(scored_k)

        ml_data_cv = reduce(DataFrame.unionByName, oof_predictions)

        # ── DML residuals ─────────────────────────────────────────────────
        ml_data_cv = ml_data_cv.withColumn(
            "y_res", F.col("post_campaign_total_order_value") - F.col("y_hat")
        )
        ml_data_cv = ml_data_cv.withColumn(
            "t_hat", vector_to_array(F.col("probability"))[1]
        )
        ml_data_cv = ml_data_cv.withColumn(
            "t_hat",
            F.when(F.col("t_hat") > 0.95, 0.95)
            .when(F.col("t_hat") < 0.05, 0.05)
            .otherwise(F.col("t_hat")),
        )
        ml_data_cv = ml_data_cv.withColumn("t_res", F.col("treatment") - F.col("t_hat"))

        # ── ATE: OLS of y_res ~ t_res ─────────────────────────────────────
        final_assembler = VectorAssembler(
            inputCols=["t_res"], outputCol="final_features", handleInvalid="skip"
        )
        final_ml_data  = final_assembler.transform(ml_data_cv)
        causal_model   = LinearRegression(
            featuresCol="final_features", labelCol="y_res",
            predictionCol="final_pred", fitIntercept=True,
            maxIter=10, tol=1e-3, solver="normal",
        )
        final_estimate = causal_model.fit(final_ml_data)

        # ── Evaluation ────────────────────────────────────────────────────
        eval_y = RegressionEvaluator(
            labelCol="post_campaign_total_order_value",
            predictionCol="y_hat", metricName="r2"
        )
        r2_o = float(eval_y.evaluate(ml_data_cv))

        eval_t = BinaryClassificationEvaluator(
            labelCol="treatment",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc_t = float(eval_t.evaluate(ml_data_cv))

        # ── Lift summary statistics ───────────────────────────────────────
        causal_summary = final_estimate.summary
        lift_value    = float(final_estimate.coefficients[0])
        lift_std_err  = float(causal_summary.coefficientStandardErrors[0])
        lift_p_value  = float(causal_summary.pValues[0])
        lift_ci_lower = lift_value - 1.96 * lift_std_err
        lift_ci_upper = lift_value + 1.96 * lift_std_err

        stats = ml_data_cv.agg(
            F.count("*").alias("total_count"),
            F.sum(F.col("treatment").cast(DoubleType())).alias("treated_count"),
            F.avg(F.when(
                F.col("treatment") == 1,
                F.col("post_campaign_total_order_value"),
            )).alias("avg_treatment_spend"),
        ).collect()[0]

        total_count         = stats["total_count"]
        treated_count       = stats["treated_count"]
        avg_treatment_spend = stats["avg_treatment_spend"] or 0.0

        expected_spend    = avg_treatment_spend - lift_value
        lift_percent      = (lift_value / expected_spend * 100) if expected_spend != 0 else 0
        lift_pct_ci_lower = (lift_ci_lower / expected_spend * 100) if expected_spend != 0 else 0
        lift_pct_ci_upper = (lift_ci_upper / expected_spend * 100) if expected_spend != 0 else 0

        # ── CATE: segment interaction terms ──────────────────────────────
        if "has_baseline_purchase" not in ml_data_cv.columns:
            ml_data_cv = ml_data_cv.withColumn(
                "has_baseline_purchase",
                F.when(F.col("days_since_last_baseline_purchase") < 366, F.lit(1))
                 .otherwise(F.lit(0))
                 .cast(IntegerType()),
            )

        # Weighted average income from raw counts for CATE segmentation
        _income_total    = reduce(lambda a, b: a + b,
                                  [F.col(f"num_hh_income_code_{i}_in_addresslink") for i in range(1, 36)])
        _income_weighted = reduce(lambda a, b: a + b,
                                  [F.lit(float(i)) * F.col(f"num_hh_income_code_{i}_in_addresslink")
                                   for i in range(1, 36)])
        _age_buckets = [
            ("age_18_24_count", 21.0), ("age_25_34_count", 29.5), ("age_35_44_count", 39.5),
            ("age_45_54_count", 49.5), ("age_55_64_count", 59.5), ("age_65_74_count", 69.5),
            ("age_75_84_count", 79.5), ("age_85_plus_count", 90.0),
        ]
        _age_total    = reduce(lambda a, b: a + b, [F.col(c) for c, _ in _age_buckets])
        _age_weighted = reduce(lambda a, b: a + b, [F.lit(m) * F.col(c) for c, m in _age_buckets])

        ml_data_cate = (
            ml_data_cv
            .withColumn("est_age",
                        F.when(_age_total > 0, _age_weighted / _age_total)
                        .otherwise(F.lit(40.0)).cast(DoubleType()))
            .withColumn("est_income_code",
                        F.when(_income_total > 0, _income_weighted / _income_total)
                        .otherwise(F.lit(18.0)).cast(DoubleType()))
            .withColumn("seg_buyer",       F.col("has_baseline_purchase").cast(DoubleType()))
            .withColumn("seg_lapsed",      F.when(F.col("days_since_last_baseline_purchase") >= 30,
                                                  F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn("seg_young",       F.when(F.col("est_age") <= 35,
                                                  F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn("seg_senior",      F.when(F.col("est_age") >= 55,
                                                  F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn("seg_high_income", F.when(F.col("est_income_code") >= 20,
                                                  F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn("t_res_x_buyer",       F.col("t_res") * F.col("seg_buyer"))
            .withColumn("t_res_x_lapsed",      F.col("t_res") * F.col("seg_lapsed"))
            .withColumn("t_res_x_young",       F.col("t_res") * F.col("seg_young"))
            .withColumn("t_res_x_senior",      F.col("t_res") * F.col("seg_senior"))
            .withColumn("t_res_x_high_income", F.col("t_res") * F.col("seg_high_income"))
        )

        cate_feature_cols = ["t_res", "t_res_x_buyer", "t_res_x_lapsed",
                             "t_res_x_young", "t_res_x_senior", "t_res_x_high_income"]
        cate_assembler = VectorAssembler(
            inputCols=cate_feature_cols, outputCol="cate_features", handleInvalid="skip"
        )
        cate_ml_data  = cate_assembler.transform(ml_data_cate)
        cate_model    = LinearRegression(
            featuresCol="cate_features", labelCol="y_res",
            predictionCol="cate_prediction", fitIntercept=True,
        )
        cate_estimate = cate_model.fit(cate_ml_data)

        coefs        = cate_estimate.coefficients
        cate_summary = cate_estimate.summary
        ate_base             = float(coefs[0])
        cate_buyer       = ate_base + float(coefs[1])
        cate_lapsed      = ate_base + float(coefs[2])
        cate_young       = ate_base + float(coefs[3])
        cate_senior      = ate_base + float(coefs[4])
        cate_high_income = ate_base + float(coefs[5])

        try:
            cate_buyer_p_value       = float(cate_summary.pValues[1])
            cate_lapsed_p_value      = float(cate_summary.pValues[2])
            cate_young_p_value       = float(cate_summary.pValues[3])
            cate_senior_p_value      = float(cate_summary.pValues[4])
            cate_high_income_p_value = float(cate_summary.pValues[5])
        except:
            cate_buyer_p_value = cate_lapsed_p_value = cate_young_p_value = -1.0
            cate_senior_p_value = cate_high_income_p_value = -1.0

        result = spark.createDataFrame(
            [(
                lift_value, lift_percent, avg_treatment_spend, expected_spend,
                total_count, treated_count,
                r2_o, auc_t,
                lift_pct_ci_lower, lift_pct_ci_upper,
                lift_ci_lower, lift_ci_upper, lift_p_value,
                ate_base, cate_buyer, cate_lapsed, cate_young, cate_senior, cate_high_income,
                cate_buyer_p_value, cate_lapsed_p_value, cate_young_p_value,
                cate_senior_p_value, cate_high_income_p_value,
            )],
            [
                "incremental_lift", "lift_percent", "avg_treatment_amount", "expected_amount",
                "total_row_count", "treated_count",
                "outcome_r2", "treatment_auc",
                "lift_pct_ci_lower", "lift_pct_ci_upper",
                "lift_ci_lower", "lift_ci_upper", "lift_p_value",
                "ate_base", "cate_existing_buyer", "cate_lapsed", "cate_young",
                "cate_senior", "cate_high_income",
                "cate_existing_buyer_p_value", "cate_lapsed_p_value", "cate_young_p_value",
                "cate_senior_p_value", "cate_high_income_p_value",
            ],
        )

        return result
