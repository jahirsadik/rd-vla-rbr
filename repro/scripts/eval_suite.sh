#!/usr/bin/env bash
# Run one full LIBERO suite (10 tasks) for an explicit checkpoint path, tag and seed, with the 10
# task processes in parallel on GPU ${RDVLA_GPU:-1}. Thin wrapper over eval_task.sh for ad-hoc runs (e.g. an
# intermediate checkpoint). The pipeline uses run_evals.sh instead.
#
# Usage: repro/scripts/eval_suite.sh <ckpt_dir> <suite_id> <src> <tag> <seed> [extra eval flags...]
# Env: EVAL_PAR (default 10), EVAL_TASKS (default "0 1 2 3 4 5 6 7 8 9"), EVAL_TRIALS (default 50)
set -uo pipefail
if [[ $# -lt 5 ]]; then sed -n '2,8p' "$0"; exit 1; fi
CKPT=$1; SUITE=$2; SRC=$3; TAG=$4; SEED=$5; shift 5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
PAR=${EVAL_PAR:-10}; TASKS=${EVAL_TASKS:-"0 1 2 3 4 5 6 7 8 9"}
OUT="repro/results/$SRC/$TAG/$SUITE/s$SEED"
START=$(date +%s)
echo "[eval_suite] $SRC/$TAG/$SUITE seed=$SEED tasks=[$TASKS] par=$PAR -> $OUT"
export CKPT SUITE SRC TAG SEED; export EXTRA_STR="$*"
echo "$TASKS" | tr ' ' '\n' | xargs -P "$PAR" -I{} bash -c '
  # shellcheck disable=SC2086
  bash "'"$SCRIPT_DIR"'/eval_task.sh" "$CKPT" "$SUITE" "$SRC" "$TAG" "$SEED" {} $EXTRA_STR'
echo "suite_elapsed_s: $(( $(date +%s) - START ))" >> "$OUT/run_info.txt"
python "$SCRIPT_DIR/aggregate.py" --one "$OUT" || true
