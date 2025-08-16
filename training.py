from typing import Tuple, Optional
from pyspark.sql import DataFrame, functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from app_logging.logger import get_logger
from exception_handling import log_exceptions
from feature_engineering import build_feature_pipeline, prepare_dataframe

logger = get_logger(__name__)


@log_exceptions(logger)
def train_model(
    spark,
    train_path: str,
    sample_fraction: float = 0.05,
    seed: int = 42,
    hash_bins: int = 2**16,
) -> Tuple[PipelineModel, dict]:
    """Train a logistic regression CTR model on Avazu with optional sampling.\n
    Args:\n
      spark: SparkSession\n
      train_path: path to train.csv or train.gz\n
      sample_fraction: fraction for downsampling during local dev (set to 1.0 for full training)\n
      hash_bins: number of hashing bins for categorical features\n
    Returns:\n
      (fitted_pipeline_model, metrics)\n
    """
    logger.info(f"Reading training data from {train_path}")
    df = spark.read.csv(train_path, header=True, inferSchema=True)
    df = prepare_dataframe(df)

    # Label must be double
    df = df.withColumn("label", F.col("click").cast("double"))

    if sample_fraction and sample_fraction < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
        logger.info(f"Sampled training data to fraction={sample_fraction}")

    # Train/validation split
    train_df, val_df = df.randomSplit([0.9, 0.1], seed=seed)

    # Class weighting to handle imbalance
    pos = train_df.filter(F.col("label") == 1.0).count()
    neg = train_df.filter(F.col("label") == 0.0).count()
    balance_ratio = neg / max(pos, 1)
    logger.info(f"Class balance: pos={pos}, neg={neg}, ratio={balance_ratio:.3f}")
    train_df = train_df.withColumn(
        "classWeightCol", F.when(F.col("label") == 1.0, balance_ratio).otherwise(1.0)
    )

    pipeline = build_feature_pipeline(hash_bins=hash_bins)

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.001,
        elasticNetParam=0.0,
        aggregationDepth=2,
        weightCol="classWeightCol",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        predictionCol="prediction",
    )

    full_pipeline = PipelineModel(stages=pipeline.getStages()).copy()
    # Fit has to be on a Pipeline, not PipelineModel, so rebuild cleanly:
    from pyspark.ml import Pipeline

    full = Pipeline(stages=[*pipeline.getStages(), lr])

    logger.info("Fitting model...")
    model = full.fit(train_df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    val_pred = model.transform(val_df)
    auc = evaluator.evaluate(val_pred)

    metrics = {"val_auc": float(auc), "train_pos": int(pos), "train_neg": int(neg)}
    logger.info(f"Validation AUC: {auc:.6f}")
    return model, metrics
