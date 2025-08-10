from pyspark.sql import functions as F, types as T
from pyspark.ml.feature import FeatureHasher, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel

AVAZU_COLUMNS = [
    "id","click","hour","C1","banner_pos","site_id","site_domain","site_category",
    "app_id","app_domain","app_category","device_id","device_ip","device_model",
    "device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"
]

CATEGORICAL_COLS = [
    "C1","banner_pos","site_id","site_domain","site_category",
    "app_id","app_domain","app_category","device_id","device_ip","device_model",
    "device_type","device_conn_type"
]

NUMERIC_COLS = ["C14","C15","C16","C17","C18","C19","C20","C21"]

DERIVED_NUMERIC_COLS = ["hour_of_day","day_of_week"]

def with_time_features(df):
    # hour is yyyymmddHH
    df = df.withColumn("hour_str", F.col("hour").cast("string"))
    # Make sure it's 10 digits
    df = df.withColumn("hour_str", F.lpad("hour_str", 10, "0"))
    df = df.withColumn("dt", F.to_timestamp(F.col("hour_str"), "yyyyMMddHH"))
    df = df.withColumn("hour_of_day", F.hour("dt").cast("int"))
    df = df.withColumn("day_of_week", F.dayofweek("dt").cast("int"))
    return df.drop("dt", "hour_str")

def build_feature_pipeline(hash_bins: int = 2**18) -> Pipeline:
    """Return a Pipeline that:
    - adds time-based features
    - hashes high-cardinality categoricals
    - assembles numeric + hashed vectors
    """
    def cast_str(cols):
        return [F.col(c).cast("string").alias(c) for c in cols]

    def prep(df):
        # Ensure required cols exist (some datasets may differ)
        for c in CATEGORICAL_COLS + NUMERIC_COLS + ["hour"]:
            if c not in df.columns:
                raise ValueError(f"Missing expected column: {c}")
        return df

    hasher = FeatureHasher(inputCols=CATEGORICAL_COLS, outputCol="hashed_cat", numFeatures=hash_bins)

    assembler = VectorAssembler(
        inputCols=["hashed_cat"] + NUMERIC_COLS + DERIVED_NUMERIC_COLS,
        outputCol="features"
    )

    pipeline = Pipeline(stages=[
        # time features via SQL Transformer (applied separately in code for clarity)
        hasher,
        assembler
    ])
    return pipeline

def prepare_dataframe(df):
    # Cast types and handle NAs
    for c in NUMERIC_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))
    for c in CATEGORICAL_COLS:
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit("missing")).otherwise(F.col(c)))
    df = with_time_features(df)
    return df
