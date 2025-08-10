from typing import Optional
from pyspark.sql import functions as F
from logging.logger import get_logger
from exception_handling import log_exceptions
from feature_engineering import prepare_dataframe

logger = get_logger(__name__)

@log_exceptions(logger)
def run_inference(model, spark, test_path: str, out_path: str = "submissions/submission.csv"):
    logger.info(f"Reading test data from {test_path}")
    df = spark.read.csv(test_path, header=True, inferSchema=True)
    df = prepare_dataframe(df)
    preds = model.transform(df).select("id", F.col("probability").getItem(1).alias("click"))
    logger.info(f"Writing predictions to {out_path}")
    preds.coalesce(1).write.mode("overwrite").option("header", True).csv(out_path)
    logger.info("Done. (Note: Spark writes a folder with a single CSV part file)")