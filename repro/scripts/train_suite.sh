#!/usr/bin/env bash
# Train one LIBERO suite on ONE GPU (RDVLA_GPU, default 1) with the repo's YAML recipe.
#
# Usage: repro/scripts/train_suite.sh <spatial|object|goal|long> [extra --key=value overrides]
# Example: repro/scripts/train_suite.sh spatial
#          RDVLA_GPU=0 RUN_NAME=z1_spatial_b64_s42 TRAIN_LOG=repro/logs/train_spatial_b64.log \
#            repro/scripts/train_suite.sh spatial --grad_accumulation_steps=4     (effective batch 64)
#
# Effective batch = batch_size 16 x grad_accumulation_steps. Single-GPU [DEV]: 16 x 2 = 32 (runbook: 2 GPUs = 64).
# A max_steps "step" is one optimizer step, so --grad_accumulation_steps=4 takes ~2x as long per step.
#
# Multi-GPU is refused: on z-1 the two GPUs are treated as separate machines (user decision, 2026-10-13).
# Tested then: NCCL P2P hangs at the first barrier, the SHM transport dies with "illegal memory access";
# only NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 passed a bare all_reduce test, never a training run.
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
LOG="${TRAIN_LOG:-repro/logs/train_${SUITE}.log}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "[train_suite] refusing multi-GPU ($CUDA_VISIBLE_DEVICES): train on one GPU, see the header" >&2; exit 2
fi
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
