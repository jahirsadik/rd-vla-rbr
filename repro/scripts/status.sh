#!/usr/bin/env bash
# One-screen status of the reproduction: GPU, training progress + loss, pipeline stage, eval results.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
echo "== $(date '+%F %T')  GPUs: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | paste -sd';')   disk: $(df -h . | awk 'NR==2{print $4" free"}')"
echo "== tmux: $(tmux list-windows -t rdvla -F '#W' 2>/dev/null | paste -sd' ' || echo 'no session')"
for s in spatial object goal long; do
  log=repro/logs/train_$s.log
  [[ -f $log ]] || continue
  prog=$(tail -c 400 "$log" | tr '\r' '\n' | grep -o "[0-9]*/[0-9]* \[[^]]*\]" | tail -1)
  loss=$(tr '\r' '\n' < "$log" | sed 's/\x1b\[[0-9;]*m//g' | grep -o "\[train step [0-9]*\] loss_value=[0-9.]*" | tail -1)
  end=$(grep "^\[train_suite\] end" "$log" | tail -1)
  echo "== train $s: ${prog:-starting}  ${loss}  ${end}"
done
echo "== checkpoints:"; ls -d outputs/*tars_*_chkpt 2>/dev/null | sed 's/^/   /' || true
echo "== pipeline (last 6):"; tail -6 repro/logs/pipeline.log 2>/dev/null | sed 's/^/   /'
[[ -f repro/GATE_spatial.txt ]] && { echo "== gate:"; sed 's/^/   /' repro/GATE_spatial.txt; }
echo "== results so far:"
python repro/scripts/aggregate.py 2>/dev/null | grep -E "^\| (Fixed|Binary|Pure|Linear)|^### " | sed 's/^/   /'
