#!/usr/bin/env bash
# Spatial gate for the effective-batch-64 retrain (z1_spatial_b64_s42, final step 39999), plus the Fixed-1
# recurrence check. Results go to repro/results/ours_b64/. One queue per GPU, one suite (10 tasks) at a time.
#
# Usage: tmux new-window -t rdvla -n gate_gpu0 'bash repro/scripts/z1_gate_b64.sh 0 2>&1 | tee -a repro/logs/z1_gate_b64_gpu0.log'
#        tmux new-window -t rdvla -n gate_gpu1 'bash repro/scripts/z1_gate_b64.sh 1 2>&1 | tee -a repro/logs/z1_gate_b64_gpu1.log'
set -uo pipefail
GPU=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RDVLA_GPU=$GPU EVAL_PAR=${EVAL_PAR:-10}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh" > /dev/null
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib.sh"
CKPT="outputs/configs+libero_spatial_no_noops+b32+lr-0.0002+lora-r64--image_aug--z1_spatial_b64_s42--39999_chkpt"
[[ -d "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }

# GPU 0 gets 4 suites, GPU 1 gets 3; seed 7 first on both.
if [[ $GPU == 0 ]]; then RUNS="fixed12:7 fixed1:7 binary_5e-4:8 fixed12:9"
else RUNS="binary_5e-4:7 fixed12:8 binary_5e-4:9"; fi

log_line "z1_gate_b64 gpu$GPU: start [$RUNS] ckpt=$CKPT git $(git rev-parse --short HEAD)"
for r in $RUNS; do
  tag=${r%%:*}; seed=${r##*:}
  # shellcheck disable=SC2046
  bash "$SCRIPT_DIR/eval_suite.sh" "$CKPT" libero_spatial ours_b64 "$tag" "$seed" $(tag_flags "$tag")
done
python "$SCRIPT_DIR/aggregate.py" --out repro/results_tables.md > /dev/null 2>&1 || true
log_line "z1_gate_b64 gpu$GPU: ALL DONE"
