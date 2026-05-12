#!/usr/bin/env bash
# run_seed_sim.sh — Launch two GPU chains for the seed-similarity baseline.
#
#   chain 1 (OH_GPU,   default 1): OpenHermes math -> OpenHermes code
#   chain 2 (TULU_GPU, default 2): Tulu       math -> Tulu       code
#
# Each chain pins one GPU through CUDA_DEVICE_ORDER=PCI_BUS_ID +
# CUDA_VISIBLE_DEVICES so the child only sees that single device.
# Python is run with -u so stdout/stderr are line-buffered and the
# log file updates in real time.
#
# Logs:    _sim_oh_chain.log, _sim_tulu_chain.log
# Stop:    bash stop_seed_sim.sh
# Status:  bash monitor_seed_sim.sh

set -e
cd "$(dirname "$0")"

PY="./.venv/bin/python -u"
SCRIPT=seed_similarity_filter.py
OH_GPU="${OH_GPU:-1}"      # nvidia-smi index
TULU_GPU="${TULU_GPU:-2}"  # nvidia-smi index

if pgrep -f "$SCRIPT" > /dev/null; then
  echo "ERROR: $SCRIPT is already running. Stop it first:"
  echo "  bash stop_seed_sim.sh"
  exit 1
fi

echo "Launching OpenHermes chain (GPU $OH_GPU)..."
nohup bash -c "export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=$OH_GPU; export PYTHONUNBUFFERED=1; \
$PY $SCRIPT --pool openhermes --seed_dir openhermes_math_oracle --out_dir openhermes_math_sim --device cuda:0 && \
$PY $SCRIPT --pool openhermes --seed_dir openhermes_code_oracle --out_dir openhermes_code_sim --device cuda:0" \
  > _sim_oh_chain.log 2>&1 < /dev/null &
disown
echo "  PID=$!  log: _sim_oh_chain.log"

echo "Launching Tulu chain (GPU $TULU_GPU)..."
nohup bash -c "export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=$TULU_GPU; export PYTHONUNBUFFERED=1; \
$PY $SCRIPT --pool tulu --seed_dir tulu_math_oracle --out_dir tulu_math_sim --device cuda:0 && \
$PY $SCRIPT --pool tulu --seed_dir tulu_code_oracle --out_dir tulu_code_sim --device cuda:0" \
  > _sim_tulu_chain.log 2>&1 < /dev/null &
disown
echo "  PID=$!  log: _sim_tulu_chain.log"

echo
echo "Monitor: tail -f _sim_oh_chain.log    OR    bash monitor_seed_sim.sh"
echo "Stop:    bash stop_seed_sim.sh"
echo
echo "Override GPUs (nvidia-smi indices):"
echo "  OH_GPU=3 TULU_GPU=4 bash run_seed_sim.sh"
