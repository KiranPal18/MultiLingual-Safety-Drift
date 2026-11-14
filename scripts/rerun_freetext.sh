#!/usr/bin/env bash
# Re-generate + re-judge the free-text evals after the stop-token fix (MCQ results are unaffected and kept).
#   PYTHONUNBUFFERED=1 setsid nohup bash scripts/rerun_freetext.sh > artifacts/logs/rerun.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python E=artifacts/logs/eval M=artifacts/models
SETS=harmful,xstest,benign_medical
gen() { [ -f $E/$1/benign_medical.jsonl ] || $PY scripts/03_generate.py --name "$@" --only $SETS; }
gen base
gen sft --adapter $M/aya-med-sft/final
gen safemix --adapter $M/aya-med-sft-safemix/final
gen resta --model $M/aya-med-resta
$PY scripts/04_judge.py base sft resta safemix
$PY scripts/07_report.py base sft resta safemix
echo "=== done"
