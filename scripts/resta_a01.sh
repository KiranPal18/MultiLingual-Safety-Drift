#!/usr/bin/env bash
# Ablation: gentler RESTA (alpha=0.1). Needs the base/sft/resta/safemix eval logs for the report.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -f artifacts/models/aya-med-resta-a0.1/config.json ] || \
  $PY scripts/05_repair_resta.py --adapter artifacts/models/aya-med-sft/final --alpha 0.1 --out artifacts/models/aya-med-resta-a0.1
[ -f artifacts/logs/eval/resta_a0.1/benign_medical.jsonl ] || \
  $PY scripts/03_generate.py --name resta_a0.1 --model artifacts/models/aya-med-resta-a0.1
$PY scripts/04_judge.py resta_a0.1
$PY scripts/07_report.py base sft resta resta_a0.1 safemix
echo "=== done"
