import os, json, pathlib
from typing import Dict, Any

def save_metrics_json(metrics: Dict[str, Any], out_dir: str, filename: str = "metrics.json"):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path
