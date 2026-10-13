#!/usr/bin/env bash
# Shared two-GPU work pool for the rest of S4a (released checkpoints, seed 7, all 31 tags x 4 suites).
# Replaces z1_queue_gpu0.sh / z1_queue_gpu1.sh, which ran one xargs batch per tag group per GPU: every
# batch ended with a straggler tail, and the GPU whose share was smaller sat idle at the end.
#
# One job list, slowest suite (Long) first. Each GPU runs EVAL_PAR workers; a worker walks the list in
# order and claims a job with an atomic mkdir under $CLAIMS, so each job runs once, on whichever GPU
# frees a slot first. Both GPUs therefore finish within about one job of each other.
#
# Takeover of a running batch: eval_task.sh processes that are already running when this starts (and
# were not started by a pool) are pre-claimed, and worker k on GPU g only starts once at most
# EVAL_PAR-1-k of them are still running on g. A GPU therefore never runs more than EVAL_PAR evals.
# Stop the old dispatcher (its xargs) first, so it cannot start new jobs.
#
# After the pass, incomplete jobs (a failed eval leaves no complete JSON) are retried once in a second pass.
#
# Usage: tmux new-window -t rdvla -n pool 'bash repro/scripts/z1_pool.sh 2>&1 | tee -a repro/logs/z1_pool.log'
#        bash repro/scripts/z1_pool.sh --dry-run     # print the job list summary and the pre-claims, run nothing
# Env:   EVAL_PAR (default 16 per GPU), POOL_GPUS (default "0 1").
set -uo pipefail
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh" > /dev/null
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib.sh"
export EVAL_PAR=${EVAL_PAR:-16}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
GPUS=${POOL_GPUS:-"0 1"}
SRC=released; SEED=7

TAUS="1e-4 2e-4 5e-4 1e-3 5e-3 1e-2"
TAGS="fixed1 fixed2 fixed4 fixed8 fixed16 fixed24 fixed32"
for kind in purekl binary lindecay binpaper; do for t in $TAUS; do TAGS+=" ${kind}_$t"; done; done

job_done() { [[ -s "repro/results/$1/$2/$3/s$4/t$5.json" ]] && grep -q '"overall_success_rate"' "repro/results/$1/$2/$3/s$4/t$5.json"; }

# Incomplete jobs, tab-separated like run_evals.sh. Long first (~2x slower per task), then the rest tag-major.
build_jobs() {
  local out=$1 tag s suite ckpt flags tid
  : > "$out"
  for pass_suites in "long" "spatial object goal"; do
    for tag in $TAGS; do
      flags=$(tag_flags "$tag") || continue
      for s in $pass_suites; do
        suite=$(suite_id "$s"); ckpt=$(ckpt_for "$SRC" "$s")
        for tid in 0 1 2 3 4 5 6 7 8 9; do
          job_done "$SRC" "$tag" "$suite" "$SEED" "$tid" && continue
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ckpt" "$suite" "$SRC" "$tag" "$SEED" "$tid" "$flags" >> "$out"
        done
      done
    done
  done
}

# PIDs of running eval_task.sh processes that no pool started (no RDVLA_POOL in their environment).
foreign_evals() {
  local p
  for p in $(pgrep -f 'repro/scripts/eval_task\.sh '); do
    [[ "$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)" == bash\ *repro/scripts/eval_task.sh\ * ]] || continue
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q '^RDVLA_POOL=' && continue
    echo "$p"
  done
}
# How many of those run on GPU $1
foreign_on_gpu() {
  local g=$1 n=0 p
  for p in $(foreign_evals); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$g" && n=$((n+1))
  done
  echo "$n"
}

worker() {  # worker <gpu> <k> <jobs> <claims>
  local g=$1 k=$2 jobs=$3 claims=$4 ckpt suite src tag seed tid flags
  while (( $(foreign_on_gpu "$g") > EVAL_PAR - 1 - k )); do sleep 20; done
  export RDVLA_GPU=$g CUDA_VISIBLE_DEVICES=$g RDVLA_POOL=1
  while IFS=$'\t' read -r -u 3 ckpt suite src tag seed tid flags; do
    mkdir "$claims/${src}__${tag}__${suite}__s${seed}__t${tid}" 2>/dev/null || continue
    job_done "$src" "$tag" "$suite" "$seed" "$tid" && continue
    # shellcheck disable=SC2086
    bash "$SCRIPT_DIR/eval_task.sh" "$ckpt" "$suite" "$src" "$tag" "$seed" "$tid" $flags < /dev/null \
      || echo "[pool] gpu$g FAILED $src/$tag/$suite/s$seed t$tid (left for the retry pass)"
  done 3< "$jobs"
}

T0=$(date +%s)
for pass in 1 2; do
  JOBS=repro/logs/z1_pool_jobs.pass$pass.tsv
  build_jobs "$JOBS"
  n=$(wc -l < "$JOBS")
  CLAIMS=repro/logs/claims/pass$pass.$(date +%Y%m%d_%H%M%S)
  pre=()
  if (( pass == 1 )); then
    for p in $(foreign_evals); do
      read -r -a a < <(tr '\0' ' ' < "/proc/$p/cmdline")   # bash eval_task.sh ckpt suite src tag seed tid flags...
      pre+=("${a[4]}__${a[5]}__${a[3]}__s${a[6]}__t${a[7]}")
    done
  fi
  if (( DRY )); then
    echo "dry-run: pass $pass would queue $n jobs ($(cut -f2 "$JOBS" | sort | uniq -c | tr '\n' ' '))"
    echo "dry-run: first job: $(head -1 "$JOBS" | cut -f2-6 | tr '\t' ' ')   last job: $(tail -1 "$JOBS" | cut -f2-6 | tr '\t' ' ')"
    echo "dry-run: ${#pre[@]} running evals would be pre-claimed; per GPU:$(for g in $GPUS; do printf ' gpu%s=%s' "$g" "$(foreign_on_gpu "$g")"; done)"
    printf '   %s\n' "${pre[@]}" | head -40
    exit 0
  fi
  (( n == 0 )) && break
  mkdir -p "$CLAIMS"
  for c in "${pre[@]}"; do mkdir -p "$CLAIMS/$c"; done
  log_line "z1_pool: pass $pass, $n jobs, GPUs [$GPUS] x $EVAL_PAR, ${#pre[@]} running evals pre-claimed, git $(git rev-parse --short HEAD)"
  for g in $GPUS; do
    for k in $(seq 0 $((EVAL_PAR - 1))); do worker "$g" "$k" "$JOBS" "$CLAIMS" & done
  done
  wait
  # Pass 2 must not start while a pre-claimed eval from before the pool is still running.
  while [[ -n "$(foreign_evals)" ]]; do sleep 30; done
  log_line "z1_pool: pass $pass finished ($(( ($(date +%s)-T0)/60 )) min so far)"
done

python "$SCRIPT_DIR/aggregate.py" --out repro/results_tables.md > /dev/null 2>&1 || true
left=$(build_jobs /dev/stdout | wc -l)
log_line "z1_pool: ALL DONE in $(( ($(date +%s)-T0)/60 )) min, $left S4a jobs still incomplete"
