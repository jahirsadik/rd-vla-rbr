#!/usr/bin/env bash
# Train one LIBERO suite with the repo's YAML recipe, on one GPU (RDVLA_GPU, default 1) or two (RDVLA_GPU=0,1).
#
# Usage: repro/scripts/train_suite.sh <spatial|object|goal|long> [extra --key=value overrides]
# Example: repro/scripts/train_suite.sh spatial
#          RDVLA_GPU=0,1 RUN_NAME=z1_spatial_b64_s42 TRAIN_LOG=repro/logs/train_spatial_b64.log \
#            repro/scripts/train_suite.sh spatial          (2 GPUs: the runbook recipe, effective batch 64)
#
# Effective batch = batch_size 16 x grad_accumulation_steps 2 x number of GPUs (one torchrun process per GPU).
# Single-GPU [DEV]: 32. A max_steps "step" is one optimizer step, so --grad_accumulation_steps=4 on one GPU also
# gives 64 but takes ~2x as long per step (z-1: 2.92 s/step, ~32 h for 40k steps).
#
# z-1 multi-GPU (2026-10-13): the GPUs sit on PCIe "NODE" with the AMD IOMMU in translated mode (DMA-FQ).
# NCCL's default P2P transport hangs at the first barrier, and its SHM transport fails with "CUDA error: an illegal
# memory access" (also with NCCL_SHM_USE_CUDA_MEMCPY=1, NCCL_CUMEM_ENABLE=0 or NCCL_CUMEM_HOST_ENABLE=0). Sockets over
# lo work: exact all_reduce sums, 350 MB in ~50 ms with 4x4 socket threads, and a 40-step 2-GPU training smoke ran clean.
# So with more than one GPU this script forces the socket transport.
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
NPROC=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
if (( NPROC > 1 )); then
  export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
fi
mkdir -p repro/logs
PORT=$(( 29500 + RANDOM % 1000 ))

echo "[train_suite] suite=$SUITE run_name=$RUN_NAME gpu=$CUDA_VISIBLE_DEVICES nproc=$NPROC log=$LOG extra=[$*]" | tee -a "$LOG"
echo "[train_suite] start $(date -Is) git=$(git rev-parse --short HEAD)" | tee -a "$LOG"

# set -e would skip the end marker if torchrun fails, and pipeline.sh would wait forever.
set +e
torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" --master-port "$PORT" --local-addr 127.0.0.1 \
  run.py --config "configs/train/rdvla_${SUITE}.yaml" --mode train \
  --use_wandb=false --run_name="$RUN_NAME" "$@" 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
set -e

echo "[train_suite] end $(date -Is) exit=${code}" | tee -a "$LOG"
exit "$code"
