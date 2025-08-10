from typing import Tuple
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from app_logging.logger import get_logger
from feature_engineering import build_feature_pipeline, prepare_dataframe

logger = get_logger(__name__)

def train_model_parametric(spark, train_path: str, *, sample_fraction: float, seed: int, hash_bins: int,
                           regParam: float, elasticNetParam: float, maxIter: int):
    """Train Logistic Regression with provided hyperparams and return (model, metrics)."""
    df = spark.read.csv(train_path, header=True, inferSchema=True)
    df = prepare_dataframe(df)
    df = df.withColumn("label", F.col("click").cast("double"))

    if sample_fraction and sample_fraction < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
        logger.info(f"Sampled training to fraction={sample_fraction}")

    train_df, val_df = df.randomSplit([0.9, 0.1], seed=seed)

    pos = train_df.filter(F.col("label") == 1.0).count()
    neg = train_df.filter(F.col("label") == 0.0).count()
    balance_ratio = neg / max(pos, 1)
    train_df = train_df.withColumn("classWeightCol", F.when(F.col("label") == 1.0, balance_ratio).otherwise(1.0))
    logger.info(f"Class balance: pos={pos}, neg={neg}, ratio={balance_ratio:.3f}")

    feat_pipe = build_feature_pipeline(hash_bins=hash_bins)

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=int(maxIter),
        regParam=float(regParam),
        elasticNetParam=float(elasticNetParam),
        weightCol="classWeightCol",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        predictionCol="prediction",
    )

    pipe = Pipeline(stages=[*feat_pipe.getStages(), lr])
    model = pipe.fit(train_df)

    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(model.transform(val_df))

    metrics = {
        "val_auc": float(auc),
        "train_pos": int(pos),
        "train_neg": int(neg),
        "hash_bins": int(hash_bins),
        "regParam": float(regParam),
        "elasticNetParam": float(elasticNetParam),
        "maxIter": int(maxIter),
        "sample_fraction": float(sample_fraction),
    }
    return model, metrics
