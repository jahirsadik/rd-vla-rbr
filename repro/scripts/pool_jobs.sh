#!/usr/bin/env bash
# Run an arbitrary list of eval_task.sh jobs on several GPUs from one shared queue.
# Each line of the job file is tab-separated: ckpt  suite  src  tag  seed  task_id  flags
# (flags may contain spaces; per-job episode counts go in flags, e.g. "--num_trials_per_task 20", last flag wins).
# Workers claim jobs with an atomic mkdir, so each job runs once, on whichever GPU frees a slot first.
# Complete jobs (JSON with overall_success_rate) are skipped, so re-running the same file resumes it.
#
# Usage: tmux new-window -t rdvla -n <name> 'bash repro/scripts/pool_jobs.sh <jobs.tsv> 2>&1 | tee -a repro/logs/<name>.log'
# Env:   EVAL_PAR (default 16 per GPU), POOL_GPUS (default "0 1"), POOL_NAME (default: job file basename).
set -uo pipefail
JOBS=$(readlink -f "$1")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh" > /dev/null
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib.sh"
export EVAL_PAR=${EVAL_PAR:-16}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
GPUS=${POOL_GPUS:-"0 1"}
NAME=${POOL_NAME:-$(basename "$JOBS" .tsv)}
CLAIMS=repro/logs/claims/$NAME.$(date +%Y%m%d_%H%M%S)
mkdir -p "$CLAIMS"

worker() {  # worker <gpu>
  local g=$1 ckpt suite src tag seed tid flags json
  export RDVLA_GPU=$g CUDA_VISIBLE_DEVICES=$g
  while IFS=$'\t' read -r -u 3 ckpt suite src tag seed tid flags; do
    [[ -z "$ckpt" || "$ckpt" == \#* ]] && continue
    mkdir "$CLAIMS/${src}__${tag}__${suite}__s${seed}__t${tid}" 2>/dev/null || continue
    json="repro/results/$src/$tag/$suite/s$seed/t$tid.json"
    [[ -s "$json" ]] && grep -q '"overall_success_rate"' "$json" && continue
    # shellcheck disable=SC2086
    bash "$SCRIPT_DIR/eval_task.sh" "$ckpt" "$suite" "$src" "$tag" "$seed" "$tid" $flags < /dev/null \
      || echo "[pool_jobs] gpu$g FAILED $src/$tag/$suite/s$seed t$tid"
  done 3< "$JOBS"
}

n=$(grep -cvE '^\s*(#|$)' "$JOBS")
T0=$(date +%s)
log_line "pool_jobs $NAME: start, $n jobs, GPUs [$GPUS] x $EVAL_PAR, git $(git rev-parse --short HEAD)"
for g in $GPUS; do
  for _ in $(seq 1 "$EVAL_PAR"); do worker "$g" & done
done
wait
log_line "pool_jobs $NAME: ALL DONE in $(( ($(date +%s)-T0)/60 )) min"
