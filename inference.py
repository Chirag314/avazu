# inference.py
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from exception_handling import log_exceptions


@log_exceptions(logger=None)
def run_inference(model, spark, test_path: str, out_path: str):
    # Read test (Spark will read .gz if you point at a .gz)
    df = spark.read.option("header", True).csv(test_path)

    # ---------- Ensure engineered features ----------
    # hour_of_day from Avazu 'hour' (yyyymmddHH or yyMMddHH) using modulo
    if "hour_of_day" not in df.columns and "hour" in df.columns:
        df = df.withColumn(
            "hour_of_day", (F.col("hour").cast("int") % F.lit(100)).cast("int")
        )

    # day_of_week without 'u' pattern:
    # Build a timestamp ts robustly; then use dayofweek(ts) => 1=Sun,...,7=Sat
    # Convert to Mon=1,...,Sun=7 as ((dow + 5) % 7) + 1
    if "day_of_week" not in df.columns and "hour" in df.columns:
        hour_str = F.lpad(F.col("hour").cast("string"), 10, "0")
        ts_try1 = F.to_timestamp(hour_str, "yyyyMMddHH")
        ts_try2 = F.to_timestamp(F.substring(hour_str, -8, 8), "yyMMddHH")
        ts = F.coalesce(ts_try1, ts_try2)
        dow = F.dayofweek(ts)  # 1=Sun,...,7=Sat
        df = df.withColumn(
            "day_of_week", (((dow + F.lit(5)) % F.lit(7)) + F.lit(1)).cast("int")
        )

    # ---------- Enforce numeric schema ----------
    numeric_int_cols = [
        "C1",
        "banner_pos",
        "device_type",
        "device_conn_type",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "hour_of_day",
        "day_of_week",
    ]
    for c in numeric_int_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("int"))

    # id as string if present
    if "id" in df.columns:
        df = df.withColumn("id", F.col("id").cast("string"))

    # ---------- Score ----------
    scored = model.transform(df)

    # probability is a Vector; take class-1 probability
    prob_pos = vector_to_array(F.col("probability"))[1].alias("click")

    if "id" not in scored.columns:
        scored = scored.withColumn("id", F.monotonically_increasing_id().cast("string"))

    preds = scored.select(F.col("id").alias("id"), prob_pos)

    # Write CSV with header
    (preds.coalesce(1).write.mode("overwrite").option("header", True).csv(out_path))
