"""Offline depth oracle (ideas Part 1, Idea 2, Stage 2A) from a probe_loops.py trajectory dump.

u_k = mean|a_k - a_K| (distance to the K-loop decode, normalized action units); oracle depth for tolerance eps:
k*(eps) = min{k >= 1 : max_{j >= k} u_j <= eps} (label-free "stays converged"). Reports, per eps: the k*
distribution, the share of k* variance explained by the S_0 draw vs by the state, and the matched fixed depth N
whose mean u_N equals the oracle's mean u_{k*} (the fair cost comparison). Also the demo-label error at k*
and at fixed N. Usage: python research/oracle_depth_offline.py research/out/dump/released_spatial.npz [...]
"""
import json
import sys

import numpy as np

EPS = (0.02, 0.01, 0.005, 0.0025)
for path in sys.argv[1:]:
    z = np.load(path)
    A = z["actions"].astype(np.float32)  # [N, S, K+1, 8, 7], index 0 = decode of S_0
    lab = z["labels"]
    task = z["task"]
    N, S, K1 = A.shape[:3]
    K = K1 - 1
    u = np.abs(A - A[:, :, K:K + 1]).mean(axis=(3, 4))  # [N, S, K+1]
    err = np.abs(A - lab[:, None, None]).mean(axis=(3, 4))  # [N, S, K+1]
    tail = np.flip(np.maximum.accumulate(np.flip(u, axis=2), axis=2), axis=2)  # max_{j>=k} u_j
    fixed_u = u.mean(axis=(0, 1))  # mean distance to final at fixed depth k
    fixed_err = err.mean(axis=(0, 1))
    out = {"dump": path, "frames": N, "seeds": S, "K": K, "fixed_mean_u": fixed_u.round(5).tolist(),
           "fixed_mean_err": fixed_err.round(5).tolist(), "oracle": {}}
    print(f"\n== {path}: {N} frames x {S} S_0 draws, K={K}")
    print(f"fixed depth  k:   " + " ".join(f"{k:>6d}" for k in (1, 2, 4, 6, 8, 10, 12, 16)))
    print(f"mean |a_k-a_K|:   " + " ".join(f"{fixed_u[k]:6.4f}" for k in (1, 2, 4, 6, 8, 10, 12, 16)))
    print(f"mean err to demo: " + " ".join(f"{fixed_err[k]:6.4f}" for k in (1, 2, 4, 6, 8, 10, 12, 16)))
    for eps in EPS:
        ok = tail[:, :, 1:] <= eps  # k = 1..K
        kstar = ok.argmax(axis=2) + 1  # [N, S]
        idx = kstar[..., None]
        u_at = np.take_along_axis(u, idx, axis=2)[..., 0]
        e_at = np.take_along_axis(err, idx, axis=2)[..., 0]
        # matched fixed depth: smallest N with mean u_N <= oracle mean u_{k*}
        target = u_at.mean()
        n_match = int(np.argmax(fixed_u[1:] <= target) + 1) if (fixed_u[1:] <= target).any() else K
        # variance decomposition of k*: within-state (across S_0) vs between-state
        within = kstar.var(axis=1).mean()
        between = kstar.mean(axis=1).var()
        per_task = {t: float(kstar[task == t].mean()) for t in sorted(set(task.tolist()))}
        q = np.percentile(kstar, [10, 50, 90])
        res = {"mean_kstar": float(kstar.mean()), "p10_50_90": q.tolist(), "mean_u_at_kstar": float(target),
               "matched_fixed_N": n_match, "loop_saving_vs_matched_fixed": 1 - kstar.mean() / n_match,
               "head_passes_oracle(2k+2)": 2 * kstar.mean() + 2, "head_passes_fixed(2N+2)": 2 * n_match + 2,
               "err_at_kstar": float(e_at.mean()), "err_fixed_matched": float(fixed_err[n_match]),
               "var_within_state_S0": float(within), "var_between_states": float(between),
               "S0_share_of_variance": float(within / (within + between)), "per_task_mean_kstar": per_task}
        out["oracle"][str(eps)] = res
        print(f"eps={eps:<7} k* mean {kstar.mean():5.2f} (p10/50/90 {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}) | matched fixed N={n_match:2d}"
              f" | loops saved {100 * res['loop_saving_vs_matched_fixed']:5.1f}% | err k* {e_at.mean():.4f} vs fixed {fixed_err[n_match]:.4f}"
              f" | S0 share of k* variance {100 * res['S0_share_of_variance']:4.1f}%")
    json.dump(out, open(path.replace(".npz", "_oracle_depth.json"), "w"), indent=1)
