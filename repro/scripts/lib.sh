#!/usr/bin/env bash
# Shared helpers for the repro scripts. Source after env.sh.

# suite short name -> LIBERO suite id
suite_id() {
  case "$1" in
    spatial) echo libero_spatial ;; object) echo libero_object ;; goal) echo libero_goal ;; long) echo libero_10 ;;
    libero_spatial|libero_object|libero_goal|libero_10) echo "$1" ;;
    *) echo "unknown suite $1" >&2; return 1 ;;
  esac
}
# LIBERO suite id -> short name
suite_short() {
  case "$1" in
    libero_spatial|spatial) echo spatial ;; libero_object|object) echo object ;; libero_goal|goal) echo goal ;; libero_10|long) echo long ;;
    *) echo "unknown suite $1" >&2; return 1 ;;
  esac
}
# Released checkpoint dir for a suite
released_ckpt() {
  case "$(suite_short "$1")" in
    spatial) echo outputs/12_24-24_24_Spatial_40k ;; object) echo outputs/12_24-24_24_Object_40k ;;
    goal) echo outputs/12_24-24_24_Goal_60k ;; long) echo outputs/12_24-24_24_Long_75k ;;
  esac
}
# Our checkpoint dir for a suite: the highest step among outputs/*--tars_<suite>_s42--<step>_chkpt.
# The finished run's last save is max_steps-1 (39999 / 39999 / 59999 / 74999), not max_steps.
ours_ckpt() {
  local s; s=$(suite_short "$1") || return 1
  local best="" beststep=-1 d step
  for d in outputs/*--tars_${s}_s42--*_chkpt; do
    [[ -d "$d" ]] || continue
    step=${d##*--}; step=${step%_chkpt}
    if (( step > beststep )); then beststep=$step; best=$d; fi
  done
  [[ -n "$best" ]] && echo "$best"
}
# Checkpoint for <src> <suite>
ckpt_for() {
  case "$1" in released) released_ckpt "$2" ;; ours) ours_ckpt "$2" ;; *) echo "unknown src $1" >&2; return 1 ;; esac
}
# Eval flags for a tag (see aggregate.py for the convention). Prints space-separated flags.
tag_flags() {
  local tag=$1
  if [[ $tag =~ ^fixed([0-9]+)$ ]]; then
    echo "--recurrence_strategy fixed --recurrent_num_iter ${BASH_REMATCH[1]}"
  elif [[ $tag =~ ^(purekl|binary|lindecay|binpaper)_(.+)$ ]]; then
    local kind=${BASH_REMATCH[1]} tau=${BASH_REMATCH[2]}
    local base="--recurrence_strategy kl_divergence --recurrence_kl_thresh $tau --recurrence_max_iter 32"
    case $kind in
      purekl)   echo "$base" ;;
      binary)   echo "$base --adaptive_exec True" ;;
      lindecay) echo "$base --use_linear_decay_horizon True" ;;
      binpaper) echo "$base --adaptive_exec True --adaptive_exec_threshold 8 --adaptive_exec_low 8 --adaptive_exec_high 4" ;;
    esac
  else
    echo "unknown tag $tag" >&2; return 1
  fi
}
# True if <src>/<tag>/<suite>/s<seed> has 10 complete task JSONs
eval_done() {
  local d="repro/results/$1/$2/$3/s$4"
  local n=0 f
  for f in "$d"/t{0..9}.json; do
    [[ -s "$f" ]] && grep -q '"overall_success_rate"' "$f" && n=$((n+1))
  done
  [[ $n -eq 10 ]]
}
log_line() { echo "[$(date '+%F %T')] $*" | tee -a repro/logs/pipeline.log; }
