#!/usr/bin/env bash
# Train one LIBERO suite on a single GPU (TARS: GPU 1) with the repo's YAML recipe.
#
# Usage: repro/scripts/train_suite.sh <spatial|object|goal|long> [extra --key=value overrides]
# Example: repro/scripts/train_suite.sh spatial
#          repro/scripts/train_suite.sh spatial --grad_accumulation_steps=4   (effective batch 64 fallback)
#
# Single-GPU [DEV]: effective batch = batch_size 16 x grad_accumulation_steps 2 = 32.
# finetune() calls dist.barrier(), so it must run under torchrun even with 1 process.
# --local-addr 127.0.0.1: torchrun's c10d rendezvous otherwise advertises the node by its FQDN; on z-1 the
# hostname `z` resolves (Tailscale MagicDNS) to a *different* tailnet machine, so the agent waits 300 s and
# dies with "client socket has timed out ... (<host>.<tailnet>.ts.net, <port>)". Harmless on TARS.
# First launch on a fresh machine also spends ~4 min JIT-compiling NCCL kernels for sm_120 (see env.sh).
set -euo pipefail
SUITE=$1; shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

RUN_NAME="${RUN_NAME:-tars_${SUITE}_s42}"
LOG="repro/logs/train_${SUITE}.log"
mkdir -p repro/logs
PORT=$(( 29500 + RANDOM % 1000 ))

echo "[train_suite] suite=$SUITE run_name=$RUN_NAME gpu=$CUDA_VISIBLE_DEVICES log=$LOG extra=[$*]" | tee -a "$LOG"
echo "[train_suite] start $(date -Is) git=$(git rev-parse --short HEAD)" | tee -a "$LOG"

# set -e would skip the end marker if torchrun fails, and pipeline.sh would wait forever.
set +e
torchrun --standalone --nnodes 1 --nproc-per-node 1 --master-port "$PORT" --local-addr 127.0.0.1 \
  run.py --config "configs/train/rdvla_${SUITE}.yaml" --mode train \
  --use_wandb=false --run_name="$RUN_NAME" "$@" 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
set -e

echo "[train_suite] end $(date -Is) exit=${code}" | tee -a "$LOG"
exit "$code"
