"""Loop-liveness probe and trajectory dump on fixed LIBERO demo frames, for one or more checkpoints.

For every frame, decodes the action chunk after each loop k = 0..K (k = 0 decodes the noise S_0) from several
S_0 draws, and records per-loop latent change and the adapter's state-branch share. Writes, per checkpoint,
research/out/probe/<name>.npz (raw arrays) and <name>.json (summary metrics):

  dead_a1_a32     mean |a_1 - a_32|           (~0 means the loop does nothing after the first step)
  gap_k           mean |a_k - a_32| for k in 0..K (convergence profile)
  err_k           mean |a_k - label| for k in 0..K (error to the demo chunk, training normalization)
  seed_std_k      std of a_k across S_0 draws (init sensitivity) at k = 1, 2, 4, 12, 32
  rho_k           ||W_S S_{k-1}|| / ||W_pre S_pre|| at k = 1, 2, 4, 12, 32 (state vs prelude branch)
  dS_k            relative per-slot state change at k = 1, 2, 4, 12, 32
  vis_ratio_1     mean |a_2 - a_1| / (relative state change at loop 2): action-visible share of latent motion

Frames: the first `--eps_per_task` demos of each of the suite's tasks, `--frames_per_ep` evenly spaced steps each
(the last 7 steps excluded, as in training). Same frames for every checkpoint.

Usage: python research/probe_loops.py --name released_spatial --ckpt outputs/12_24-24_24_Spatial_40k [--ckpt ...]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from research.rdvla_tools import (Features, LoopRunner, eval_obs, iter_rlds_episodes, load_policy,
                                  normalized_chunk_labels, vlm_features_batch)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", required=True)
ap.add_argument("--name", action="append", required=True)
ap.add_argument("--dataset", default="libero_spatial_no_noops")
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--eps_per_task", type=int, default=4)
ap.add_argument("--frames_per_ep", type=int, default=8)
ap.add_argument("--K", type=int, default=32)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--anderson", action="store_true", help="also record Anderson vs Picard at budgets 1..12")
ap.add_argument("--out", default="research/out/probe")
ap.add_argument("--max_tasks", type=int, default=10)
args = ap.parse_args()
assert len(args.ckpt) == len(args.name)
os.makedirs(args.out, exist_ok=True)

# ---- select frames once -----------------------------------------------------------------------------------------
t0 = time.time()
per_task = defaultdict(int)
frames = []  # (episode_idx, t, obs, instruction, raw actions of the episode)
n_tasks_seen = set()
for ei, ep in iter_rlds_episodes(args.dataset):
    task = ep["instruction"]
    n_tasks_seen.add(task)
    if per_task[task] >= args.eps_per_task:
        if len(n_tasks_seen) >= args.max_tasks and all(per_task[k] >= args.eps_per_task for k in n_tasks_seen):
            break
        continue
    per_task[task] += 1
    T = len(ep["action"])
    for t in np.linspace(0, T - 8, args.frames_per_ep).round().astype(int):
        frames.append((ei, int(t), eval_obs(ep, int(t)), task, ep["action"]))
print(f"[probe] {len(frames)} frames from {len(per_task)} tasks in {time.time() - t0:.0f}s", flush=True)

PROBE_KS = [1, 2, 4, 12, 32]

for ckpt, name in zip(args.ckpt, args.name):
    t1 = time.time()
    pol = load_policy(ckpt, suite=args.suite)
    stats = pol.model.norm_stats[pol.cfg.unnorm_key]["action"]
    labels = np.stack([normalized_chunk_labels(a, t, stats) for (_, t, _, _, a) in frames])  # [N, 8, 7]
    # batched VLM pass per demo (frames of one demo share the prompt); numerics are batch-16-like (bf16), see
    # research/check_batch_features.py. Features are kept per frame, in frame order.
    feats = [None] * len(frames)
    by_ep = defaultdict(list)
    for j, fr in enumerate(frames):
        by_ep[fr[0]].append(j)
    for ep_idx, js in by_ep.items():
        fb = vlm_features_batch(pol, [frames[j][2] for j in js], frames[js[0]][3])
        for q, j in enumerate(js):
            feats[j] = Features(fb.h_a[q:q + 1], fb.h_t[q:q + 1], fb.p[q:q + 1], fb.layers)
    print(f"[probe] {name}: features in {time.time() - t1:.0f}s", flush=True)
    runner = LoopRunner(pol.m)
    N, K, S = len(frames), args.K, args.seeds
    A = np.zeros((N, S, K + 1, 8, 7), np.float32)
    dS = np.zeros((N, S, K), np.float32)
    rho = np.zeros((N, S, K), np.float32)
    for s in range(S):
        gen = torch.Generator(device="cuda").manual_seed(1000 + s)
        for i in range(0, N, args.batch):
            f = Features.cat(feats[i:i + args.batch])
            tr = runner.trajectory(f, K, generator=gen)
            A[i:i + args.batch, s] = tr["actions"].cpu().numpy()
            dS[i:i + args.batch, s] = tr["dS"].cpu().numpy()
            rho[i:i + args.batch, s] = tr["rho"].cpu().numpy()
    extra = {}
    if args.anderson:
        budgets = list(range(1, 13))
        ref = A[:, :, K]  # Picard after K loops, per seed
        and_gap = np.zeros((len(budgets), N, S), np.float32)
        pic_gap = np.zeros((len(budgets), N, S), np.float32)
        for bi, b in enumerate(budgets):
            pic_gap[bi] = np.abs(A[:, :, b] - ref).mean(axis=(2, 3))
            for s in range(S):
                gen = torch.Generator(device="cuda").manual_seed(1000 + s)  # same S_0 as the Picard run
                for i in range(0, N, args.batch):
                    f = Features.cat(feats[i:i + args.batch])
                    ctx = runner.prepare(f)
                    s0 = runner.init_state(f.h_a.shape[0], ctx, gen)
                    out = runner.anderson(f, budget=b, state0=s0).float().cpu().numpy()
                    and_gap[bi, i:i + args.batch, s] = np.abs(out - ref[i:i + args.batch, s]).mean(axis=(1, 2))
        extra = {"anderson_gap_to_picard32": and_gap.mean(axis=(1, 2)).round(5).tolist(),
                 "picard_gap_to_picard32": pic_gap.mean(axis=(1, 2)).round(5).tolist(), "budgets": budgets}
        np.savez_compressed(os.path.join(args.out, f"{name}_anderson.npz"), and_gap=and_gap, pic_gap=pic_gap)

    gap = np.abs(A - A[:, :, K:K + 1]).mean(axis=(3, 4))  # [N, S, K+1]
    err = np.abs(A - labels[:, None, None]).mean(axis=(3, 4))  # [N, S, K+1]
    seed_std = A.std(axis=1).mean(axis=(2, 3))  # [N, K+1]
    da = np.abs(A[:, :, 2] - A[:, :, 1]).mean(axis=(2, 3))  # [N, S]
    summary = {
        "name": name, "ckpt": ckpt, "frames": N, "seeds": S, "K": K,
        "dead_a1_a32": float(gap[:, :, 1].mean()),
        "gap_k": gap.mean(axis=(0, 1)).round(5).tolist(),
        "err_k": err.mean(axis=(0, 1)).round(5).tolist(),
        "seed_std_k": {k: float(seed_std[:, k].mean()) for k in PROBE_KS},
        "rho_k": {k: float(rho[:, :, k - 1].mean()) for k in PROBE_KS},
        "dS_k": {k: float(dS[:, :, k - 1].mean()) for k in PROBE_KS},
        "vis_ratio_1": float((da / np.maximum(dS[:, :, 1], 1e-12)).mean()),
        "per_task_dead_a1_a32": {task: float(gap[[j for j, fr in enumerate(frames) if fr[3] == task], :, 1].mean())
                                 for task in sorted(per_task)},
        **extra,
    }
    np.savez_compressed(os.path.join(args.out, f"{name}.npz"), actions=A.astype(np.float16), dS=dS, rho=rho,
                        labels=labels, episode=np.array([fr[0] for fr in frames]), t=np.array([fr[1] for fr in frames]),
                        task=np.array([fr[3] for fr in frames]))
    json.dump(summary, open(os.path.join(args.out, f"{name}.json"), "w"), indent=1)
    print(f"[probe] {name}: dead_a1_a32={summary['dead_a1_a32']:.4f} err_k(1,4,12,32)="
          f"{[summary['err_k'][k] for k in (1, 4, 12, 32)]} rho_1={summary['rho_k'][1]:.3f} "
          f"seed_std(1,32)={summary['seed_std_k'][1]:.4f},{summary['seed_std_k'][32]:.4f} "
          f"({time.time() - t1:.0f}s)", flush=True)
    del pol, feats
    torch.cuda.empty_cache()
