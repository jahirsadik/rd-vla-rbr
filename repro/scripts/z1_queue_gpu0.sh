#!/usr/bin/env bash
# z-1 work queue for GPU 0 (2026-10-13). Runs GPU-serially on GPU 0, in parallel with z1_queue_gpu1.sh on GPU 1.
# The two queues never share a (tag, suite) set, so no task job is scheduled twice.
#
#   1. Diagnostics of OUR Spatial checkpoint (handoff Section 6.2): fixed 1/2/4/32, pure-KL 5e-4/1e-4, midpoint 20k.
#   2. S4a Table II on the RELEASED checkpoints, suites spatial + object, all 31 tags.
#   3. S4a released goal: lindecay + binpaper groups (GPU 1 does goal fixed/purekl/binary).
#
# Usage: tmux new -s rdvla -n evals_gpu0 'bash repro/scripts/z1_queue_gpu0.sh 2>&1 | tee -a repro/logs/z1_queue_gpu0.log'
set -uo pipefail
export RDVLA_GPU=0
export EVAL_PAR=${EVAL_PAR:-16}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib.sh"

TAUS="1e-4 2e-4 5e-4 1e-3 5e-3 1e-2"
FIXED="fixed1 fixed2 fixed4 fixed8 fixed16 fixed24 fixed32"
KLBIN="$(for t in $TAUS; do printf 'purekl_%s binary_%s ' "$t" "$t"; done)"
LIN="$(for t in $TAUS; do printf 'lindecay_%s ' "$t"; done)"
BINP="$(for t in $TAUS; do printf 'binpaper_%s ' "$t"; done)"
T0=$(date +%s)
log_line "z1_queue_gpu0: start (GPU $CUDA_VISIBLE_DEVICES, EVAL_PAR=$EVAL_PAR, OMP=$OMP_NUM_THREADS, git $(git rev-parse --short HEAD))"

# 1. Diagnostics on ours/spatial (seed 7)
log_line "z1_queue_gpu0: [1/3] diagnostics on ours/spatial"
bash "$SCRIPT_DIR/run_evals.sh" ours spatial 7 "fixed1 fixed2 fixed4 fixed32"
bash "$SCRIPT_DIR/run_evals.sh" ours spatial 7 "purekl_5e-4 purekl_1e-4"
MID=$(ls -d outputs/*--tars_spatial_s42--20000_chkpt 2>/dev/null | head -1)
if [[ -n "$MID" ]]; then
  EVAL_PAR=10 bash "$SCRIPT_DIR/eval_suite.sh" "$MID" libero_spatial ours_20k fixed12 7 --recurrence_strategy fixed --recurrent_num_iter 12
else
  log_line "z1_queue_gpu0: midpoint 20000 checkpoint not found, skipping"
fi
log_line "z1_queue_gpu0: diagnostics done ($(( ($(date +%s)-T0)/60 )) min so far)"

# 2. S4a released, spatial + object, all tags
log_line "z1_queue_gpu0: [2/3] S4a released spatial+object"
bash "$SCRIPT_DIR/run_evals.sh" released "spatial object" 7 "$FIXED"
bash "$SCRIPT_DIR/run_evals.sh" released "spatial object" 7 "$KLBIN"
bash "$SCRIPT_DIR/run_evals.sh" released "spatial object" 7 "$LIN"
bash "$SCRIPT_DIR/run_evals.sh" released "spatial object" 7 "$BINP"

# 3. S4a released goal: lindecay + binpaper (the rest of goal runs on GPU 1)
log_line "z1_queue_gpu0: [3/3] S4a released goal lindecay+binpaper"
bash "$SCRIPT_DIR/run_evals.sh" released goal 7 "$LIN"
bash "$SCRIPT_DIR/run_evals.sh" released goal 7 "$BINP"

python "$SCRIPT_DIR/aggregate.py" --out repro/results_tables.md > /dev/null 2>&1 || true
log_line "z1_queue_gpu0: ALL DONE in $(( ($(date +%s)-T0)/60 )) min"
