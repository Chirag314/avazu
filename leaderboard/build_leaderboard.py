import os, csv, argparse
import wandb

def build_leaderboard(entity: str, project: str, tag: str, out_csv: str):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    rows = []
    for r in runs:
        if tag and tag not in (r.tags or []):
            continue
        cfg = r.config or {}
        summary = r.summary or {}
        rows.append({
            "run_id": r.id,
            "name": r.name,
            "val_auc": float(summary.get("val_auc", float("nan"))),
            "hash_bins": cfg.get("hash_bins"),
            "regParam": cfg.get("regParam"),
            "elasticNetParam": cfg.get("elasticNetParam"),
            "maxIter": cfg.get("maxIter"),
            "sample_fraction": cfg.get("sample_fraction"),
            "created_at": str(r.created_at),
        })
    rows.sort(key=lambda x: (-(x["val_auc"] if x["val_auc"]==x["val_auc"] else -1), x["created_at"]))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
        else:
            f.write("run_id,name,val_auc\n")
    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--tag", default="avazu_sweep")
    ap.add_argument("--out", default="leaderboard/leaderboard.csv")
    args = ap.parse_args()
    build_leaderboard(args.entity, args.project, args.tag, args.out)
