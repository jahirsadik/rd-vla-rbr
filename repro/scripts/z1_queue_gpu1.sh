#!/usr/bin/env bash
# z-1 work queue for GPU 1 (2026-10-13). Runs GPU-serially on GPU 1, in parallel with z1_queue_gpu0.sh on GPU 0.
# The two queues never share a (tag, suite) set, so no task job is scheduled twice.
#
#   1. S4a Table II on the RELEASED checkpoints, suite long (libero_10), all 31 tags (Long is ~2x slower per task).
#   2. S4a released goal: fixed + purekl/binary groups (GPU 0 does goal lindecay/binpaper).
#
# Usage: tmux new-window -t rdvla -n evals_gpu1 'bash repro/scripts/z1_queue_gpu1.sh 2>&1 | tee -a repro/logs/z1_queue_gpu1.log'
set -uo pipefail
export RDVLA_GPU=1
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
log_line "z1_queue_gpu1: start (GPU $CUDA_VISIBLE_DEVICES, EVAL_PAR=$EVAL_PAR, OMP=$OMP_NUM_THREADS, git $(git rev-parse --short HEAD))"

# 1. S4a released long, all tags
log_line "z1_queue_gpu1: [1/2] S4a released long"
bash "$SCRIPT_DIR/run_evals.sh" released long 7 "$FIXED"
bash "$SCRIPT_DIR/run_evals.sh" released long 7 "$KLBIN"
bash "$SCRIPT_DIR/run_evals.sh" released long 7 "$LIN"
bash "$SCRIPT_DIR/run_evals.sh" released long 7 "$BINP"

# 2. S4a released goal: fixed + purekl/binary
log_line "z1_queue_gpu1: [2/2] S4a released goal fixed+purekl/binary"
bash "$SCRIPT_DIR/run_evals.sh" released goal 7 "$FIXED"
bash "$SCRIPT_DIR/run_evals.sh" released goal 7 "$KLBIN"

python "$SCRIPT_DIR/aggregate.py" --out repro/results_tables.md > /dev/null 2>&1 || true
log_line "z1_queue_gpu1: ALL DONE in $(( ($(date +%s)-T0)/60 )) min"
