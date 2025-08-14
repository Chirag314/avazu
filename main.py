import os
import argparse
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel  # used when loading saved model

from app_logging.logger import get_logger
from exception_handling import log_exceptions
from training import train_model
from inference import run_inference

logger = get_logger(__name__)


# --- Windows Spark bootstrap (avoid HADOOP_HOME error) ---
def ensure_windows_hadoop() -> None:
    """Configure HADOOP_HOME/winutils on Windows to prevent Spark errors."""
    if os.name != "nt":  # only on Windows
        return
    hadoop_home = os.environ.get("HADOOP_HOME") or r"C:\hadoop"
    winutils = Path(hadoop_home) / "bin" / "winutils.exe"
    if not winutils.exists():
        print(
            "⚠️ winutils.exe not found. Create C:\\hadoop\\bin and place winutils.exe there."
        )
        return

    os.environ.setdefault("HADOOP_HOME", hadoop_home)
    os.environ.setdefault("hadoop.home.dir", hadoop_home)
    os.environ.setdefault("SPARK_LOCAL_DIRS", r"C:\tmp\spark")
    Path(r"C:\tmp\spark").mkdir(parents=True, exist_ok=True)
    Path(r"C:\tmp\hive").mkdir(parents=True, exist_ok=True)
    print(f"✅ Using winutils at: {winutils}")


def spark_session(
    app_name: str = "AvazuPySpark", driver_mem: str = "6g"
) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config(
            "spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", driver_mem)
        )
        .config(
            "spark.sql.shuffle.partitions",
            os.environ.get("SPARK_SHUFFLE_PARTITIONS", "200"),
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


@log_exceptions(logger)
def maybe_download_kaggle(data_dir: str) -> None:
    """Download Avazu competition files with Kaggle CLI if they don't exist.
    Requires KAGGLE_USERNAME and KAGGLE_KEY env vars configured previously.
    """
    train_gz = os.path.join(data_dir, "train.gz")
    test_gz = os.path.join(data_dir, "test.gz")
    _sample_gz = os.path.join(data_dir, "sampleSubmission.gz")  # not used by code path

    if os.path.exists(train_gz) and os.path.exists(test_gz):
        logger.info("Kaggle data already present.")
        return

    os.makedirs(data_dir, exist_ok=True)
    logger.info("Attempting to download dataset from Kaggle...")
    cmd = f'kaggle competitions download -c avazu-ctr-prediction -p "{data_dir}"'
    os.system(cmd)
    # Spark can read .gz directly; no unzip needed


def main() -> None:
    ensure_windows_hadoop()  # safe to call on all platforms

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing Kaggle files"
    )
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=float(os.environ.get("TRAIN_SAMPLE_FRACTION", "0.05")),
    )
    parser.add_argument(
        "--hash_bins", type=int, default=int(os.environ.get("HASH_BINS", str(2**18)))
    )
    args = parser.parse_args()

    # If neither flag provided, do both
    if not args.train and not args.infer:
        args.train = True
        args.infer = True

    data_dir = args.data_dir
    maybe_download_kaggle(data_dir)

    spark = spark_session()

    train_path = os.path.join(data_dir, "train.gz")
    test_path = os.path.join(data_dir, "test.gz")

    model = None
    if args.train:
        model, metrics = train_model(
            spark,
            train_path,
            sample_fraction=args.sample_fraction,
            hash_bins=args.hash_bins,
        )
        logger.info(f"Metrics: {metrics}")

        # Save model
        model_dir = "artifacts/model"
        model.write().overwrite().save(model_dir)
        logger.info(f"Saved model to {model_dir}")

    if args.infer:
        if model is None:
            model_dir = "artifacts/model"
            logger.info(f"Loading model from {model_dir}")
            model = PipelineModel.load(model_dir)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = f"submissions/submission_{ts}"
        os.makedirs("submissions", exist_ok=True)
        run_inference(model, spark, test_path=test_path, out_path=out_dir)
        logger.info(f"Submission written to {out_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
