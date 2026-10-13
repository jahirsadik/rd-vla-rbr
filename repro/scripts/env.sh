#!/usr/bin/env bash
# Source this from anywhere: `source <project>/rd-vla-rbr/repro/scripts/env.sh`
# Sets up the rdvla conda env and all env vars needed for training/eval.
# All paths are derived from this file's location. Project layout:
#   <PROJECT>/                  (TARS: ~/Documents/Projects/rdvla-stuff; z-1: /media/jahir/8tb-data/jahir-vla/rdvla-stuff)
#     rd-vla-rbr/               this repo
#     LIBERO/                   simulator clone
#     miniforge3/               conda install; env at miniforge3/envs/rdvla
#     markdown-files/           plan, runbook, notes, paper
#     hf-cache/                 HF_HOME (timm/Qwen weights); .libero/ LIBERO config; .cache/ pip + XDG caches
# Everything the project touches stays under <PROJECT>; nothing is written to $HOME.
# GPU policy: set RDVLA_GPU=<physical index> before sourcing. Default 1.
#   TARS: GPU 1 only (GPU 0 belonged to another user).  z-1 (2026-10-13): both GPUs are ours; use
#   RDVLA_GPU=0 and RDVLA_GPU=1 in two separate shells/tmux windows to run on both.

_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RDVLA_ROOT="$(cd "$_ENV_SH_DIR/../.." && pwd)"          # .../rd-vla-rbr
export RDVLA_PROJECT="$(cd "$RDVLA_ROOT/.." && pwd)"           # .../rdvla-stuff
export RDVLA_CONDA="$RDVLA_PROJECT/miniforge3"
export RDVLA_ENV="$RDVLA_CONDA/envs/rdvla"
export LIBERO_ROOT="$RDVLA_PROJECT/LIBERO"

# Keep every cache/config inside the project (user rule: nothing in $HOME)
export HF_HOME="$RDVLA_PROJECT/hf-cache"                       # Hugging Face hub cache (timm DINOv2/SigLIP, Qwen2.5)
export LIBERO_CONFIG_PATH="$RDVLA_PROJECT/.libero"             # LIBERO reads $LIBERO_CONFIG_PATH/config.yaml (default ~/.libero)
export XDG_CACHE_HOME="$RDVLA_PROJECT/.cache"                  # fontconfig, torch hub, ... (default ~/.cache)
export MPLCONFIGDIR="$RDVLA_PROJECT/.cache/matplotlib"         # matplotlib config+cache (default ~/.config/matplotlib)
export PIP_CACHE_DIR="$RDVLA_PROJECT/.cache/pip"
export CONDA_PKGS_DIRS="$RDVLA_CONDA/pkgs"                     # ignore any ~/.condarc redirecting pkgs elsewhere
# CUDA PTX JIT cache (default ~/.nv/ComputeCache). NCCL 2.26 (cu12 wheel) ships no sm_120 SASS, so the driver
# JIT-compiles every NCCL kernel on first use: ~minutes of one thread at 100 % CPU with no output, which looks
# like a hang (TARS attempts 1-2, z-1 2026-10-13). Keep the cache in the project and large enough to never evict.
export CUDA_CACHE_PATH="$RDVLA_PROJECT/.cache/nv-compute-cache"
export CUDA_CACHE_MAXSIZE=4294967296
mkdir -p "$HF_HOME" "$LIBERO_CONFIG_PATH" "$PIP_CACHE_DIR" "$CUDA_CACHE_PATH"

# Make sure nothing from an older install location leaks in
unset MAMBA_EXE MAMBA_ROOT_PREFIX
# shellcheck disable=SC1091
source "$RDVLA_CONDA/etc/profile.d/conda.sh"
conda activate "$RDVLA_ENV"

cd "$RDVLA_ROOT" || return 1
export PYTHONPATH="$RDVLA_ROOT"

# File-descriptor limit. robosuite/MuJoCo leak ~53 fds (Panda .obj mesh files) per hard env.reset(); a 50-episode
# task eval needs ~2,700. With the Ubuntu default soft limit of 1024 every eval process spins at 100 % CPU forever
# on its 19th episode (z-1 2026-10-13: 32 processes, both GPUs idle; inherited 1024 from a pre-existing tmux server).
ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || ulimit -Sn 65536 2>/dev/null || true

# GPU pinning (physical id must match for CUDA and EGL; robosuite checks this)
export CUDA_VISIBLE_DEVICES="${RDVLA_GPU:-1}"
export MUJOCO_EGL_DEVICE_ID="${RDVLA_GPU:-1}"

# Headless rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Misc
export HF_HUB_DISABLE_TELEMETRY=1
# hf-xet's native downloader stalled (spinning thread, main thread waiting) on the first
# training launch while fetching Qwen/timm weights. Use the plain HTTP path instead.
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER   # hf_transfer is not installed in this env
# NCCL: single-process DDP; keep bootstrap on loopback (tailscale0/docker0 exist on TARS and z-1)
export NCCL_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
# Keep TensorFlow (data loader only) off the GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
