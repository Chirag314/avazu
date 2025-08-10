import os
import wandb
from main import spark_session, maybe_download_kaggle
from training_wandb import train_model_parametric
from inference import run_inference
from wandb_utils import save_metrics_json

def main():
    project = os.getenv("WANDB_PROJECT", "pyspark-avazu")
    entity = os.getenv("WANDB_ENTITY")  # optional
    tags = ["avazu_sweep", "pyspark"]
    run = wandb.init(project=project, entity=entity, tags=tags)

    cfg = wandb.config
    data_dir = os.getenv("DATA_DIR", "data")
    maybe_download_kaggle(data_dir)

    spark = spark_session(app_name="AvazuWandB", driver_mem=os.getenv("SPARK_DRIVER_MEMORY", "6g"))
    train_path = os.path.join(data_dir, "train.gz")
    test_path  = os.path.join(data_dir, "test.gz")

    model, metrics = train_model_parametric(
        spark,
        train_path=train_path,
        sample_fraction=float(cfg.get("sample_fraction", 0.05)),
        seed=int(cfg.get("seed", 42)),
        hash_bins=int(cfg.get("hash_bins", 262144)),
        regParam=float(cfg.get("regParam", 0.1)),
        elasticNetParam=float(cfg.get("elasticNetParam", 0.0)),
        maxIter=int(cfg.get("maxIter", 20)),
    )
    wandb.log(metrics)

    out_dir = f"submissions/wandb_{run.id}"
    os.makedirs("submissions", exist_ok=True)
    run_inference(model, spark, test_path=test_path, out_path=out_dir)

    art = wandb.Artifact(f"submission_{run.id}", type="submission")
    art.add_dir(out_dir)
    wandb.log_artifact(art)

    save_metrics_json(metrics, out_dir)

    spark.stop()
    wandb.finish()

if __name__ == "__main__":
    main()