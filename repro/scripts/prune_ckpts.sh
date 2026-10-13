#!/usr/bin/env bash
# Delete intermediate training checkpoints for a run, keeping the final (highest step) one and the
# half-way one (sanity). ~3.1 GB each.
#
# Usage: repro/scripts/prune_ckpts.sh <run_dir_prefix> [--dry-run]
#   e.g. repro/scripts/prune_ckpts.sh "outputs/configs+libero_spatial_no_noops+b32+lr-0.0002+lora-r64--image_aug--tars_spatial_s42"
# Matches <prefix>--<step>_chkpt directories.
set -euo pipefail
PREFIX=$1; DRY=${2:-}
DIRS=(); for d in "${PREFIX}"--*_chkpt; do [[ -d "$d" ]] && DIRS+=("$d"); done
if [[ ${#DIRS[@]} -eq 0 ]]; then echo "no checkpoints match ${PREFIX}--*_chkpt"; exit 0; fi
step_of() { local s=${1##*--}; echo "${s%_chkpt}"; }
final=-1; for d in "${DIRS[@]}"; do s=$(step_of "$d"); (( s > final )) && final=$s; done
half=-1; best=999999999
for d in "${DIRS[@]}"; do s=$(step_of "$d"); diff=$(( s - final/2 )); diff=${diff#-}; if (( diff < best )); then best=$diff; half=$s; fi; done
echo "[prune] ${#DIRS[@]} checkpoints; keeping final=$final half=$half"
for d in "${DIRS[@]}"; do
  s=$(step_of "$d")
  if [[ "$s" == "$final" || "$s" == "$half" ]]; then continue; fi
  if [[ "$DRY" == "--dry-run" ]]; then echo "[prune] would delete $d ($(du -sh "$d" | cut -f1))"; else echo "[prune] deleting $d"; rm -rf "$d"; fi
done
