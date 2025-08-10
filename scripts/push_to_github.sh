#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <https://github.com/<you>/<repo>.git>"
  exit 1
fi

REMOTE="$1"
git init
git add .
git commit -m "Initial commit: PySpark Avazu pipeline"
git branch -M main
git remote add origin "$REMOTE" || git remote set-url origin "$REMOTE"
git push -u origin main