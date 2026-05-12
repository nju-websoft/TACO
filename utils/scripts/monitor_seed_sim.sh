#!/usr/bin/env bash
# monitor_seed_sim.sh — One-shot status snapshot of the two chains.

cd "$(dirname "$0")"

echo "=== Running processes ==="
pgrep -fa "seed_similarity_filter.py" || echo "  (none)"

echo
echo "=== OpenHermes chain (last 12 lines) ==="
[ -f _sim_oh_chain.log ] && tail -12 _sim_oh_chain.log || echo "  (no log yet)"

echo
echo "=== Tulu chain (last 12 lines) ==="
[ -f _sim_tulu_chain.log ] && tail -12 _sim_tulu_chain.log || echo "  (no log yet)"

echo
echo "=== GPU snapshot ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

echo
echo "=== Output sizes ==="
for d in openhermes_math_sim openhermes_code_sim tulu_math_sim tulu_code_sim; do
  p="dataset/$d/data.json"
  if [ -f "$p" ]; then
    n=$(./.venv/bin/python -c "import json; print(len(json.load(open('$p'))))" 2>/dev/null)
    sz=$(du -h "$p" | cut -f1)
    echo "  $d: $n records, $sz"
  else
    echo "  $d: (not yet produced)"
  fi
done

echo
echo "=== Cache files ==="
ls -lh dataset/_emb_cache/ 2>/dev/null | tail -n +2 || echo "  (no cache dir)"
