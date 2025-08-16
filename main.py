import os
import argparse
from datetime import datetime
from pathlib import Path
import tempfile
import sys

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel  # used when loading saved model

from app_logging.logger import get_logger
from exception_handling import log_exceptions
from training import train_model
from inference import run_inference

logger = get_logger(__name__)

# ======================================================================
# Windows Hadoop bootstrap: ensure winutils.exe + hadoop.dll are present
# ======================================================================

HADOOP_VERSION = "3.3.6"  # works with Spark 3.x in most PySpark wheels
HADOOP_BASE = r"C:\hadoop"
HADOOP_BIN = Path(HADOOP_BASE) / "bin"
WINUTILS_URL = f"https://github.com/cdarlint/winutils/raw/master/hadoop-{HADOOP_VERSION}/bin/winutils.exe"
HADOOP_DLL_URL = f"https://github.com/cdarlint/winutils/raw/master/hadoop-{HADOOP_VERSION}/bin/hadoop.dll"


def _download_file(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())


def ensure_hadoop_binaries() -> None:
    """Make sure C:\\hadoop\\bin has winutils.exe and hadoop.dll.
    If missing, fetch from cdarlint/winutils (Hadoop {HADOOP_VERSION}).
    """
    if os.name != "nt":
        return
    winutils = HADOOP_BIN / "winutils.exe"
    hadoop_dll = HADOOP_BIN / "hadoop.dll"

    missing = []
    if not winutils.exists():
        missing.append(("winutils.exe", WINUTILS_URL, winutils))
    if not hadoop_dll.exists():
        missing.append(("hadoop.dll", HADOOP_DLL_URL, hadoop_dll))

    if missing:
        logger.warning(
            "Some Hadoop Windows binaries are missing; attempting to download..."
        )
        for name, url, dest in missing:
            try:
                _download_file(url, dest)
            except Exception as e:
                logger.error(f"Failed to download {name} from {url}: {e}")
                logger.error(
                    "You can also download manually from: "
                    f"https://github.com/cdarlint/winutils/tree/master/hadoop-{HADOOP_VERSION}/bin "
                    "and place files in C:\\hadoop\\bin"
                )
                raise

    # Add env and DLL search path
    os.environ.setdefault("HADOOP_HOME", HADOOP_BASE)
    os.environ.setdefault("hadoop.home.dir", HADOOP_BASE)
    try:
        os.add_dll_directory(str(HADOOP_BIN))  # Python 3.8+
    except Exception as e:
        logger.warning(f"Could not add DLL dir {HADOOP_BIN}: {e}")

    # Preload hadoop.dll (fail early if incompatible)
    try:
        from ctypes import WinDLL

        WinDLL(str(HADOOP_BIN / "hadoop.dll"))
        print(f"✅ hadoop.dll loaded from {HADOOP_BIN}")
    except Exception as e:
        print(
            "❌ Cannot load hadoop.dll. Ensure it matches your Spark Hadoop version and is 64-bit."
        )
        print(f"   Looked in: {HADOOP_BIN}")
        raise


def ensure_windows_hadoop() -> None:
    """Create temp dirs + ensure binaries; keep for compatibility."""
    if os.name != "nt":
        return
    # Ensure binaries first (download if needed)
    ensure_hadoop_binaries()

    # Ensure temp dirs
    Path(r"C:\tmp\spark").mkdir(parents=True, exist_ok=True)
    Path(r"C:\tmp\hive").mkdir(parents=True, exist_ok=True)
    print("✅ Windows Hadoop bootstrap done.")


# Run bootstrap early
ensure_windows_hadoop()


def spark_session(
    app_name: str = "AvazuPySpark", driver_mem: str = "12g"
) -> SparkSession:
    # robust local dirs
    local_tmp = (
        Path(r"C:\tmp\spark")
        if os.name == "nt"
        else Path(tempfile.gettempdir()) / "spark"
    )
    local_tmp.mkdir(parents=True, exist_ok=True)
    warehouse_dir = local_tmp / "warehouse"
    warehouse_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config(
            "spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", driver_mem)
        )
        # size/time safety:
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")
        # Windows/Hadoop friction reducers:
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.hadoop.native.lib", "false")  # older key some builds still honor
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config(
            "spark.hadoop.fs.AbstractFileSystem.file.impl",
            "org.apache.hadoop.fs.local.LocalFs",
        )
        .config("mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("mapreduce.fileoutputcommitter.cleanup-failures.ignored", "true")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config(
            "spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored",
            "true",
        )
        # local dirs:
        .config("spark.sql.warehouse.dir", str(warehouse_dir))
        .config("spark.local.dir", str(local_tmp))
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

    if os.path.exists(train_gz) and os.path.exists(test_gz):
        logger.info("Kaggle data already present.")
        return

    os.makedirs(data_dir, exist_ok=True)
    logger.info("Attempting to download dataset from Kaggle...")
    cmd = f'kaggle competitions download -c avazu-ctr-prediction -p "{data_dir}"'
    code = os.system(cmd)
    if code != 0:
        logger.error(
            "Kaggle download failed (non-zero exit). Ensure KAGGLE_USERNAME/KAGGLE_KEY are set."
        )
    # Spark can read .gz directly; no unzip needed


def _abs_uri(p: Path) -> str:
    """Convert a Path to a Windows-safe absolute file:/// URI."""
    return "file:///" + str(p.resolve()).replace("\\", "/")


def main() -> None:
    # ensure_windows_hadoop() already called above, safe on all platforms

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

    # Quick FS sanity: Parquet write (isolates FS issues early)
    tmp_check = Path("artifacts/tmp_check")
    tmp_check_uri = _abs_uri(tmp_check)
    spark.range(5).write.mode("overwrite").parquet(tmp_check_uri)
    logger.info(f"✅ Parquet write test OK at {tmp_check_uri}")

    train_path = os.path.join(data_dir, "train.gz")
    test_path = os.path.join(data_dir, "test.gz")

    # Build absolute model URI
    model_root = Path("artifacts/model")
    model_root.mkdir(parents=True, exist_ok=True)
    model_uri = _abs_uri(model_root)

    saved_ok = False
    model = None

    if args.train:
        model, metrics = train_model(
            spark,
            train_path,
            sample_fraction=args.sample_fraction,
            hash_bins=args.hash_bins,
        )
        logger.info(f"Metrics: {metrics}")

        # Try save; continue even if save fails on Windows
        try:
            if not isinstance(model, PipelineModel):
                raise TypeError(f"Expected PipelineModel, got {type(model)}")
            model.write().overwrite().save(model_uri)
            logger.info(f"Saved model to {model_uri}")
            saved_ok = True
        except Exception as e:
            logger.error(
                "❌ Spark model save failed; proceeding without saving.", exc_info=True
            )

    if args.infer:
        if not saved_ok:
            # Try loading from disk; if unavailable, use in-memory model
            if model is None:
                logger.info(f"Loading model from {model_uri}")
                model = PipelineModel.load(model_uri)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"submissions/submission_{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)

        run_inference(
            model, spark, test_path=test_path, out_path=str(out_dir.resolve())
        )
        logger.info(f"Submission written to {out_dir.resolve()}")

    spark.stop()


if __name__ == "__main__":
    main()
