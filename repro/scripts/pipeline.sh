#!/usr/bin/env bash
# GPU-serial orchestrator for the TARS reproduction. GPU 1 only (env.sh pins it); never GPU 0.
# Run inside tmux: `bash repro/scripts/pipeline.sh`. Idempotent: every stage skips work that is
# already complete, so it can be re-run after an interruption.
#
# !! Bash reads a running script from disk incrementally. Never edit this file while an instance is
# !! running: stop the orchestrator (not training), edit, restart.
#
# Stage plan (2026-10-12: Object/Goal/Long retraining is SKIPPED for now)
#  S0  wait for the Spatial training run (launched separately) to finish; keep final + midpoint ckpt.
#      "Finished" = the training process has exited AND the log has "Training complete at step"
#      (or train_suite.sh's "[train_suite] end ... exit=0"). It is NOT a folder name: finetune.py's
#      last save is step max_steps-1 (Spatial --39999_chkpt), because log_step is 0-based. ours_ckpt
#      picks the highest-step dir, so 39999 is treated as final.
#  S1  gate: eval ours/Spatial fixed12 + binary_5e-4 (seed 7); pass if fixed12 >= 92.0 - 3.0 = 89.0 %
#  S2  tier 1: released ckpts, fixed12 + binary_5e-4, 4 suites x seeds 7 8 9; then ours/Spatial seeds 8 9
#  S3  SKIPPED. No Object/Goal/Long training. If S1 failed: stop after S2 and report (no recipe change).
#  S4  Table II sweep (fixed N; pure-KL / binary / linear-decay / paper-direction-binary thresholds) on the
#      RELEASED checkpoints, 4 suites, seed 7; then the same on ours/Spatial if time allows.
#  S5  aggregate -> repro/results_tables.md
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib.sh"
mkdir -p repro/logs
export EVAL_PAR=${EVAL_PAR:-16}   # parallel eval processes on GPU 1 (~5 GB, ~3 threads each)

PAPER_SPATIAL_FIXED12=92.0
GATE_MARGIN=3.0
T1_TAGS="fixed12 binary_5e-4"
ALL="spatial object goal long"
TAUS="1e-4 2e-4 5e-4 1e-3 5e-3 1e-2"
FIXED_SWEEP="fixed1 fixed2 fixed4 fixed8 fixed16 fixed24 fixed32"
KL_SWEEP="$(for t in $TAUS; do printf 'purekl_%s binary_%s ' "$t" "$t"; done)"
LD_SWEEP="$(for t in $TAUS; do printf 'lindecay_%s ' "$t"; done)"
BP_SWEEP="$(for t in $TAUS; do printf 'binpaper_%s ' "$t"; done)"

training_alive() {  # <suite>
  pgrep -f "run.py --config configs/train/rdvla_$1.yaml" > /dev/null
}
training_finished() {  # <suite> -> 0 if done OK, 1 if still running/unknown, 2 if failed
  local log="repro/logs/train_$1.log"
  [[ -f "$log" ]] || return 1
  if grep -q "^\[train_suite\] end.*exit=0" "$log"; then return 0; fi
  if grep -q "^\[train_suite\] end.*exit=[1-9]" "$log"; then return 2; fi
  # Fallback (old train_suite.sh had set -e and could skip the end line): the "Training complete"
  # line is printed BEFORE the final save, so also require the process to be gone and a ckpt dir.
  if ! training_alive "$1" && grep -q "Training complete at step" "$log" && [[ -n "$(ours_ckpt "$1")" ]]; then
    return 0
  fi
  if ! training_alive "$1" && ! grep -q "Training complete at step" "$log"; then return 2; fi
  return 1
}
wait_for_training() {  # <suite>
  log_line "S0: waiting for training '$1' (process exit + 'Training complete' / end marker in repro/logs/train_$1.log)"
  while true; do
    training_finished "$1"; local rc=$?
    if [[ $rc -eq 0 ]]; then log_line "S0: training '$1' finished; final ckpt = $(ours_ckpt "$1")"; return 0; fi
    if [[ $rc -eq 2 ]]; then log_line "S0: training '$1' FAILED or died (see repro/logs/train_$1.log)"; return 1; fi
    sleep 120
  done
}
prune() {  # <suite>
  local ds; case $1 in spatial) ds=libero_spatial_no_noops;; object) ds=libero_object_no_noops;; goal) ds=libero_goal_no_noops;; long) ds=libero_10_no_noops;; esac
  bash "$SCRIPT_DIR/prune_ckpts.sh" "outputs/configs+${ds}+b32+lr-0.0002+lora-r64--image_aug--tars_${1}_s42" 2>&1 | tee -a repro/logs/pipeline.log
  log_line "disk after prune: $(df -h / | awk 'NR==2{print $4" free"}')"
}
sr_of() {  # <src> <tag> <suite_id> <seed> -> success % or empty
  python - "$@" <<'EOF'
import sys, glob, json
src, tag, suite, seed = sys.argv[1:5]
eps = succ = 0
for f in glob.glob(f"repro/results/{src}/{tag}/{suite}/s{seed}/t*.json"):
    d = json.load(open(f))
    for stats in d.get("tasks", {}).values():
        eps += len(stats); succ += sum(1 for s in stats if s["success"])
print(f"{100*succ/eps:.1f}" if eps else "")
EOF
}
tables() { python "$SCRIPT_DIR/aggregate.py" --out repro/results_tables.md > /dev/null 2>&1 && log_line "tables -> repro/results_tables.md"; }

# ---------------- S0 ----------------
log_line "pipeline start (git $(git rev-parse --short HEAD), EVAL_PAR=$EVAL_PAR, GPU=$CUDA_VISIBLE_DEVICES, S3 skipped)"
wait_for_training spatial || { log_line "S0: stopping"; exit 1; }
prune spatial
log_line "S0: ours/spatial checkpoint = $(ours_ckpt spatial)"

# ---------------- S1 gate ----------------
log_line "S1: gate evals on ours/spatial (seed 7)"
bash "$SCRIPT_DIR/run_evals.sh" ours spatial 7 "$T1_TAGS"
SR=$(sr_of ours fixed12 libero_spatial 7); SRB=$(sr_of ours binary_5e-4 libero_spatial 7)
GATE_OK=0
if [[ -n "$SR" ]] && python -c "import sys; sys.exit(0 if float('$SR') >= $PAPER_SPATIAL_FIXED12 - $GATE_MARGIN else 1)"; then GATE_OK=1; fi
{
  echo "date: $(date -Is)"
  echo "ours/spatial fixed12 seed7: ${SR:-n/a} (paper 92.0, threshold $(python -c "print($PAPER_SPATIAL_FIXED12-$GATE_MARGIN)"))"
  echo "ours/spatial binary_5e-4 seed7: ${SRB:-n/a} (paper 88.6)"
  echo "gate_passed: $GATE_OK"
  echo "recipe: 1 GPU, batch 16 x grad_accum 2 = effective 32 (runbook 2-GPU default was 64), lr 2e-4, 40k steps"
} > repro/GATE_spatial.txt
log_line "S1: gate result: fixed12=${SR:-n/a} binary_5e-4=${SRB:-n/a} -> passed=$GATE_OK"

# ---------------- S2 tier 1 ----------------
log_line "S2: tier 1 released (seeds 7 8 9) + ours/spatial seeds 8 9"
bash "$SCRIPT_DIR/run_evals.sh" released "$ALL" "7 8 9" "$T1_TAGS"
bash "$SCRIPT_DIR/run_evals.sh" ours spatial "8 9" "$T1_TAGS"
tables

# ---------------- S3 skipped ----------------
if [[ $GATE_OK -ne 1 ]]; then
  log_line "S1 GATE FAILED: ours/spatial fixed12=${SR:-n/a} < 89.0. Reporting, not changing the recipe. First suspect: effective batch 32 on 1 GPU vs the runbook's 64 (fallback would be --grad_accumulation_steps=4)."
  log_line "S3 skipped by plan (no Object/Goal/Long retraining). Stopping here for a human decision."
  exit 2
fi
log_line "S3 skipped by plan (no Object/Goal/Long retraining)."

# ---------------- S4 Table II sweeps ----------------
log_line "S4a: Table II on RELEASED ckpts, 4 suites, seed 7: fixed sweep"
bash "$SCRIPT_DIR/run_evals.sh" released "$ALL" 7 "$FIXED_SWEEP"
log_line "S4a: pure-KL + binary sweeps"
bash "$SCRIPT_DIR/run_evals.sh" released "$ALL" 7 "$KL_SWEEP"
log_line "S4a: linear decay sweep"
bash "$SCRIPT_DIR/run_evals.sh" released "$ALL" 7 "$LD_SWEEP"
log_line "S4a: paper-direction binary sweep"
bash "$SCRIPT_DIR/run_evals.sh" released "$ALL" 7 "$BP_SWEEP"
tables
log_line "S4b: Table II on ours/spatial, seed 7"
bash "$SCRIPT_DIR/run_evals.sh" ours spatial 7 "$FIXED_SWEEP"
bash "$SCRIPT_DIR/run_evals.sh" ours spatial 7 "$KL_SWEEP"
bash "$SCRIPT_DIR/run_evals.sh" ours spatial 7 "$LD_SWEEP"
bash "$SCRIPT_DIR/run_evals.sh" ours spatial 7 "$BP_SWEEP"

# ---------------- S5 ----------------
tables
log_line "pipeline DONE"
