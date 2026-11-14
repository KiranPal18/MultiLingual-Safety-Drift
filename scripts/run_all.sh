#!/usr/bin/env bash
# Full pipeline after data build (01). Each stage is skipped if its output already exists,
# so the script can be re-run after a crash. Run from the project root:
#   setsid nohup bash scripts/run_all.sh > artifacts/logs/run_all.log 2>&1 &
#   tail -f artifacts/logs/run_all.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
E=artifacts/logs/eval
M=artifacts/models
stage() { echo; echo "=== [$(date +%H:%M)] $*"; }

while pgrep -f "02_train_sft.py --out $M/aya-med-sft$" >/dev/null; do  # a manual training run is still going
  echo "waiting for running SFT to finish ..."; sleep 120; done

stage "1. plain medical SFT (AlignTune)"
[ -f $M/aya-med-sft/final/adapter_config.json ] || $PY scripts/02_train_sft.py --out $M/aya-med-sft

stage "2. generate: base, sft, aligned answers for safety pool"
[ -f $E/base/benign_medical.jsonl ] || $PY scripts/03_generate.py --name base
[ -f $E/sft/benign_medical.jsonl ]  || $PY scripts/03_generate.py --name sft --adapter $M/aya-med-sft/final
[ -f $E/aligned_safety_pool/safety_pool.jsonl ] || $PY scripts/03_generate.py --name aligned_safety_pool --only safety_pool

stage "3. judge with Qwen3Guard"
$PY scripts/04_judge.py base sft aligned_safety_pool

stage "4. safety-mix data (CuratorKIT) + safety-mixed SFT (AlignTune)"
[ -f artifacts/datasets/med_sft_safetymix.jsonl ] || $PY scripts/06_build_safety_mix.py build
[ -f $M/aya-med-sft-safemix/final/adapter_config.json ] || \
  $PY scripts/02_train_sft.py --data artifacts/datasets/med_sft_safetymix.jsonl --out $M/aya-med-sft-safemix

stage "5. RESTA weight-space repair (SafeTune, CPU)"
[ -f $M/aya-med-resta/config.json ] || $PY scripts/05_repair_resta.py --adapter $M/aya-med-sft/final --alpha 0.25 --out $M/aya-med-resta

stage "6. generate + judge repaired models"
[ -f $E/safemix/benign_medical.jsonl ] || $PY scripts/03_generate.py --name safemix --adapter $M/aya-med-sft-safemix/final
[ -f $E/resta/benign_medical.jsonl ]   || $PY scripts/03_generate.py --name resta --model $M/aya-med-resta
$PY scripts/04_judge.py safemix resta

stage "7. report (AuditKIT)"
$PY scripts/07_report.py base sft resta safemix
stage "done"
