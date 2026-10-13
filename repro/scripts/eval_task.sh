#!/usr/bin/env bash
# Evaluate ONE LIBERO task (50 episodes by default) for one checkpoint / tag / seed.
# This is the unit of parallelism: run_evals.sh fans these out with xargs -P on GPU ${RDVLA_GPU:-1}.
# Idempotent: skips if the JSON already has "overall_success_rate".
#
# Usage: repro/scripts/eval_task.sh <ckpt_dir> <suite_id> <src> <tag> <seed> <task_id> [extra eval flags...]
# Output tree: repro/results/<src>/<tag>/<suite>/s<seed>/{t<task>.json, logs/t<task>.out, logs/*.txt, run_info.txt}
set -uo pipefail
if [[ $# -lt 6 ]]; then sed -n '2,8p' "$0"; exit 1; fi
CKPT=$1; SUITE=$2; SRC=$3; TAG=$4; SEED=$5; TID=$6; shift 6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh" > /dev/null
TRIALS=${EVAL_TRIALS:-50}
# Keep each eval process from grabbing all cores when many run at once.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}

OUT="repro/results/$SRC/$TAG/$SUITE/s$SEED"
JSON="$OUT/t$TID.json"
mkdir -p "$OUT/logs"
if [[ -s "$JSON" ]] && grep -q '"overall_success_rate"' "$JSON"; then
  echo "[eval_task] $SRC/$TAG/$SUITE/s$SEED t$TID already complete"; exit 0
fi
if [[ ! -f "$OUT/run_info.txt" ]]; then
  { echo "date: $(date -Is)"; echo "ckpt: $CKPT"; echo "suite: $SUITE"; echo "seed: $SEED"; echo "trials: $TRIALS"
    echo "gpu: ${CUDA_VISIBLE_DEVICES}"; echo "git: $(git rev-parse --short HEAD)"; echo "extra_flags: $*"; } > "$OUT/run_info.txt"
fi
t0=$(date +%s)
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CKPT" \
  --task_suite_name "$SUITE" --task_id "$TID" \
  --num_trials_per_task "$TRIALS" --seed "$SEED" \
  --save_videos False \
  --local_log_dir "$OUT/logs" --run_id_note "t$TID" \
  --json_log_file "$JSON" \
  "$@" > "$OUT/logs/t$TID.out" 2>&1
code=$?
dt=$(( $(date +%s) - t0 ))
errs=$(grep -c "Episode error" "$OUT"/logs/*t$TID.txt 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
sr=$(python - "$JSON" <<'EOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1])); print(f"{100*d.get('overall_success_rate', 0):.0f}%")
EOF
)
echo "[eval_task] $SRC/$TAG/$SUITE/s$SEED t$TID done in ${dt}s exit=$code sr=${sr:-n/a} episode_errors=$errs"
echo "t$TID elapsed_s: $dt exit: $code episode_errors: $errs" >> "$OUT/run_info.txt"
exit $code
