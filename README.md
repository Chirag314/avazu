# PySpark CTR (Avazu) – End-to-End Kaggle Pipeline

This repo trains a **PySpark** model on Kaggle's **Avazu Click-Through Rate Prediction** competition and creates a Kaggle-ready `submission.csv`.
It runs locally in VS Code or automatically via GitHub Actions.

> Dataset: Avazu CTR Prediction (Kaggle competition). You'll need to accept the rules on Kaggle first.

---

## Quickstart (Local / VS Code)

1. **Prereqs**
   - Install Java 8+ (JDK 11 recommended).
   - Python 3.10+
   - (Optional) Create a virtual env.

2. **Install deps**
   ```bash
   pip install -r requirements.txt
   ```

3. **Kaggle credentials (for auto-download)**
   - Set env vars (or create `.env`):
     ```bash
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_key
     ```
   - Or put `kaggle.json` under `~/.kaggle/` (ignored by git).

4. **Run pipeline (sampled for speed)**
   ```bash
   python main.py --data_dir data --sample_fraction 0.02
   ```
   For full training remove `--sample_fraction` or set it to `1.0`.

5. **Where are the files?**
   - Model: `artifacts/model/`
   - Submissions: `submissions/submission_YYYYmmdd_HHMMSS/part-*.csv` (rename to `submission.csv` for Kaggle upload).

---

## Project Structure

```
.
├── artifacts/
├── data/                      # (created at runtime) Kaggle files live here: train.gz, test.gz, sampleSubmission.gz
├── logging/
│   └── logger.py
├── submissions/
├── feature_engineering.py
├── training.py
├── inference.py
├── exception_handling.py
├── main.py
├── pipeline.ipynb             # Notebook version of the pipeline
├── requirements.txt
├── .github/workflows/run_pipeline.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## Notebook (`pipeline.ipynb`)

The notebook mirrors the code: Spark init, EDA, feature engineering, training, evaluation, and inference.

---

## GitHub Actions (CI)

This workflow:
- Installs Java + PySpark
- Downloads Kaggle data using repository **secrets** (`KAGGLE_USERNAME` & `KAGGLE_KEY`)
- Runs the pipeline
- Uploads the generated `submission_*.csv` as a build artifact
- Commits the CSV into `submissions/` on the default branch

Add repository secrets in **Settings → Secrets and variables → Actions**.

---

## Weights & Biases (optional)

If `WANDB_API_KEY` is present, you can log metrics and artifacts (extend `training.py` accordingly). For example:

```python
import wandb
wandb.init(project="pyspark-avazu")
wandb.log(metrics)  # e.g., AUC
```

For README embeds or sweep links, paste your W&B run/board URLs here.

---

## Push this repo to GitHub from your desktop

```bash
# Set your repo variables
export GIT_REMOTE=https://github.com/<you>/<repo>.git

git init
git add .
git commit -m "Initial commit: PySpark Avazu pipeline"
git branch -M main
git remote add origin "$GIT_REMOTE"
git push -u origin main
```

---

## Notes

- Spark can read `.gz` files directly. Training on the full Avazu dataset is **large**; start with a small `--sample_fraction` then scale up.
- Tune `HASH_BINS` (default `2**18`) and `spark.sql.shuffle.partitions` for performance.
- This repo **does not** track raw data in git; GitHub Actions downloads data each run.