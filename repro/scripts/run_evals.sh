#!/usr/bin/env bash
# Parallel eval-table runner. Builds the list of (seed, tag, suite, task_id) jobs that are not yet
# complete and runs them EVAL_PAR at a time on GPU ${RDVLA_GPU:-1} via eval_task.sh (one process per task).
# Jobs are ordered seed-major, then tag, then suite, so complete rows appear early.
#
# Usage: repro/scripts/run_evals.sh <src: released|ours> "<suites>" "<seeds>" "<tags>"
#   suites: short names (spatial object goal long) or LIBERO ids
#   e.g.  run_evals.sh released "spatial object goal long" "7 8 9" "fixed12 binary_5e-4"
#         run_evals.sh ours "spatial" "7" "fixed1 fixed2 fixed4 fixed8 fixed16 fixed24 fixed32"
# Env: EVAL_PAR (default 16; each eval process uses ~5 GB of GPU memory and ~3 CPU threads).
set -uo pipefail
SRC=$1; SUITES=$2; SEEDS=$3; TAGS=$4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib.sh"
PAR=${EVAL_PAR:-16}

JOBS=$(mktemp)
njobs=0; nskip=0
for seed in $SEEDS; do
  for tag in $TAGS; do
    for s in $SUITES; do
      suite=$(suite_id "$s") || continue
      ckpt=$(ckpt_for "$SRC" "$s")
      if [[ -z "$ckpt" || ! -d "$ckpt" ]]; then log_line "run_evals: no checkpoint for $SRC/$s, skipping"; continue; fi
      flags=$(tag_flags "$tag") || continue
      for tid in 0 1 2 3 4 5 6 7 8 9; do
        if [[ -s "repro/results/$SRC/$tag/$suite/s$seed/t$tid.json" ]] \
           && grep -q '"overall_success_rate"' "repro/results/$SRC/$tag/$suite/s$seed/t$tid.json"; then
          nskip=$((nskip+1)); continue
        fi
        # tab-separated so flags (which contain spaces) survive
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ckpt" "$suite" "$SRC" "$tag" "$seed" "$tid" "$flags" >> "$JOBS"
        njobs=$((njobs+1))
      done
    done
  done
done
log_line "run_evals: $SRC seeds=[$SEEDS] tags=[$TAGS] suites=[$SUITES] -> $njobs task jobs ($nskip already done), EVAL_PAR=$PAR"
if [[ $njobs -gt 0 ]]; then
  START=$(date +%s)
  # Each line -> one eval_task.sh invocation; flags are re-split on spaces inside the child.
  tr '\t' '\n' < "$JOBS" | xargs -d '\n' -n 7 -P "$PAR" bash -c '
    ckpt=$1; suite=$2; src=$3; tag=$4; seed=$5; tid=$6; flags=$7
    # shellcheck disable=SC2086
    exec bash "'"$SCRIPT_DIR"'/eval_task.sh" "$ckpt" "$suite" "$src" "$tag" "$seed" "$tid" $flags
  ' _ 2>&1 | tee -a "repro/logs/evals_${SRC}.log" | grep --line-buffered "\[eval_task\]"
  log_line "run_evals: finished $njobs jobs in $(( $(date +%s) - START ))s"
fi
rm -f "$JOBS"
python "$SCRIPT_DIR/aggregate.py" --out repro/results_tables.md > /dev/null 2>&1 || true
