# Weights & Biases: Sweeps + Leaderboard (Add-on)

This add-on keeps the original package **unchanged** and adds optional **W&B** sweeps + a generated **leaderboard**.

## Secrets needed (GitHub → Settings → Secrets and variables → Actions)

- `KAGGLE_USERNAME`, `KAGGLE_KEY`
- `WANDB_API_KEY` (required)
- `WANDB_ENTITY` (your W&B org/username)
- `WANDB_PROJECT` (e.g., `pyspark-avazu`)

## Run a sweep from GitHub Actions

Open the **Actions** tab → **Run W&B Sweep** → **Run workflow**.

It will:
1. Install deps & login to Kaggle + W&B
2. Create a sweep from `sweeps/sweep.yaml`
3. Launch `wandb agent` to run multiple trials of `sweeps/train_wandb_entrypoint.py`
4. Build and commit `leaderboard/leaderboard.csv`
5. Upload the best submissions as artifacts

## Local single-run with W&B logging (optional)

```bash
export WANDB_API_KEY=... WANDB_ENTITY=<you> WANDB_PROJECT=pyspark-avazu
python sweeps/train_wandb_entrypoint.py
```

## README Embed Snippets (copy into README.md when ready)

- **Project link badge**
  ```md
  [![W&B Project – pyspark-avazu](https://img.shields.io/badge/W%26B-Project-blue)](https://wandb.ai/<ENTITY>/<PROJECT>)
  ```

- **Top runs (Report)**
  ```md
  [![Top Runs (W&B Report)](https://wandb.ai/<ENTITY>/<PROJECT>/reports/<REPORT_SLUG>.png)](https://wandb.ai/<ENTITY>/<PROJECT>/reports/<REPORT_SLUG>)
  ```