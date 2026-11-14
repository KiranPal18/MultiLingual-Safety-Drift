#!/usr/bin/env bash
# Run the pipeline on a rented GPU over SSH (e.g. a RunPod pod with "SSH over exposed TCP").
#   export REMOTE=root@<pod-ip> PORT=<ssh-port>        # from the pod's "Connect" tab
#   bash scripts/remote.sh push     # copy code + data + finished results to the server
#   bash scripts/remote.sh setup    # create the Python env on the server (once, ~10 min)
#   bash scripts/remote.sh login    # Hugging Face login on the server (you paste your own token)
#   bash scripts/remote.sh run [script]   # start run_all.sh (or another script) on the server (survives disconnect)
#   bash scripts/remote.sh status   # progress
#   bash scripts/remote.sh pull [logs]    # bring results/models (or only artifacts/logs) back into artifacts/
set -euo pipefail
cd "$(dirname "$0")/.."
: "${REMOTE:?set REMOTE=root@<ip>}" "${PORT:?set PORT=<ssh port>}"
RDIR=${RDIR:-/workspace/lexsi}            # /workspace = RunPod persistent volume
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new "$REMOTE")
RSYNC=(rsync -az --info=progress2 -e "ssh -p $PORT")
ENV="export HF_HOME=/workspace/hf_cache PYTHONUNBUFFERED=1; cd $RDIR"

case "${1:-}" in
push)
  "${SSH[@]}" "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync); mkdir -p $RDIR"
  "${RSYNC[@]}" --exclude .venv --exclude data/raw --exclude runs --exclude third_party/CircuitKIT \
    --exclude 'checkpoint-*' --exclude '__pycache__' --exclude '.git' \
    --exclude 'artifacts/models/aya-med-resta*' ./ "$REMOTE:$RDIR/" ;;  # merged 6 GB models: rebuilt on demand
setup)
  "${SSH[@]}" "set -e; $ENV
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\$HOME/.local/bin:\$PATH
    [ -x .venv/bin/python ] || uv venv -p 3.12 .venv
    uv pip install -p .venv torch==2.13.0 ${TORCH_INDEX:+--index-url $TORCH_INDEX}
    uv pip install -p .venv -r scripts/remote/requirements.lock
    uv pip install -p .venv --no-deps -e third_party/aligntune -e third_party/AuditKIT -e third_party/CuratorKIT -e third_party/SafeTune
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    .venv/bin/python -c 'import torch, bitsandbytes; print(\"torch\", torch.__version__, \"cuda ok:\", torch.cuda.is_available())'
    .venv/bin/hf auth whoami || echo 'NOT LOGGED IN yet: run  bash scripts/remote.sh login'" ;;
login)  # token is stored under HF_HOME on the volume, where the pipeline looks for it
  ssh -t -p "$PORT" "$REMOTE" "$ENV; .venv/bin/hf auth login && .venv/bin/hf auth whoami" ;;
run)
  "${SSH[@]}" "$ENV; export GEN_BS=${GEN_BS:-64} JUDGE_BS=${JUDGE_BS:-32} TRAIN_BS=${TRAIN_BS:-4}; setsid nohup bash ${2:-scripts/run_all.sh} > artifacts/logs/run_all.log 2>&1 < /dev/null & sleep 2; pgrep -af 'bash scripts/'" ;;
status)
  "${SSH[@]}" "$ENV; pgrep -af '0[2-7]_.*\.py' || echo '(no stage running)'; tr -d '\000' < artifacts/logs/run_all.log | grep -vE 'CUDACaching|Loading weights' | tail -15; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv" ;;
pull)
  if pgrep -f scripts/run_all.sh >/dev/null; then echo "local run_all.sh is running - stop it first (pkill -f run_all.sh; pkill -f 0[2-7]_)"; exit 1; fi
  if [ "${2:-}" = logs ]; then "${RSYNC[@]}" "$REMOTE:$RDIR/artifacts/logs/" artifacts/logs/
  else "${RSYNC[@]}" --exclude 'checkpoint-*' "$REMOTE:$RDIR/artifacts/" artifacts/; fi ;;
*) sed -n 2,10p "$0"; exit 1 ;;
esac
