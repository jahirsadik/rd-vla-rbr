"""Idea 1 Stage 1A diagnostics (ideas Part 1 §1.2) from a probe_loops.py dump: is a_1 -> a_k directed progress
or wandering, and does the code's stopping rule fire early ("fake convergence")?

Per frame and S_0 draw (actions a_k, k = 0..K, normalized units; e_k = mean|a_k - label|; reference a_K):
  stop_k(delta)    first k >= 2 with mean((a_k - a_{k-1})^2) < delta  (exactly the eval's kl_divergence rule, max 32)
  fake(delta)      stop fires but mean((a_stop - a_K)^2) > 10 * delta  (the action would still have moved)
  monotone_frac    fraction of k in 1..11 with e_{k+1} < e_k
  progress_cos     mean over k = 1..8 of cos(a_{k+1} - a_k, a_K - a_k)
  path_eff_12      |a_12 - a_1| / sum_{k=1..11} |a_{k+1} - a_k|
Usage: python research/convergence_diagnostics.py research/out/dump/released_spatial.npz [...]
"""
import json
import sys

import numpy as np

DELTAS = (1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2)
for path in sys.argv[1:]:
    z = np.load(path)
    A = z["actions"].astype(np.float32)  # [N, S, K+1, 8, 7]
    lab = z["labels"]
    N, S, K1 = A.shape[:3]
    K = K1 - 1
    flat = A.reshape(N, S, K1, -1)
    e = np.abs(A - lab[:, None, None]).mean(axis=(3, 4))  # [N, S, K+1]
    step_mse = ((flat[:, :, 1:] - flat[:, :, :-1]) ** 2).mean(axis=3)  # [N, S, K]; index j = change from k=j to j+1
    out = {"dump": path, "frames": N, "seeds": S, "stop_rule": {}}
    print(f"\n== {path}")
    for d in DELTAS:
        # change between k-1 and k is step_mse[..., k-1]; rule checks k >= 2
        ok = step_mse[:, :, 1:] < d  # k = 2..K
        fired = ok.any(axis=2)
        k_stop = np.where(fired, ok.argmax(axis=2) + 2, min(K, 32))
        a_stop = np.take_along_axis(flat, k_stop[..., None, None], axis=2)[:, :, 0]
        dist2 = ((a_stop - flat[:, :, K]) ** 2).mean(axis=2)
        fake = (dist2 > 10 * d) & fired
        res = {"mean_stop_k": float(k_stop.mean()), "fake_rate": float(fake.mean()),
               "mean_abs_to_final_at_stop": float(np.abs(a_stop - flat[:, :, K]).mean()),
               "err_at_stop": float(np.take_along_axis(e, k_stop[..., None], axis=2).mean())}
        out["stop_rule"][f"{d:g}"] = res
        print(f"delta={d:<7g} mean stop k {res['mean_stop_k']:5.2f} | fake-convergence {100 * res['fake_rate']:5.2f}% "
              f"| |a_stop - a_K| {res['mean_abs_to_final_at_stop']:.4f} | demo err at stop {res['err_at_stop']:.4f}")
    mono = (e[:, :, 2:13] < e[:, :, 1:12]).mean()
    d1 = flat[:, :, 2:10] - flat[:, :, 1:9]
    to_final = flat[:, :, K:K + 1] - flat[:, :, 1:9]
    cos = (d1 * to_final).sum(-1) / (np.linalg.norm(d1, axis=-1) * np.linalg.norm(to_final, axis=-1) + 1e-12)
    path_len = np.linalg.norm(flat[:, :, 2:13] - flat[:, :, 1:12], axis=-1).sum(axis=2)
    eff = np.linalg.norm(flat[:, :, 12] - flat[:, :, 1], axis=-1) / np.maximum(path_len, 1e-12)
    out.update({"monotone_frac_k1_12": float(mono), "progress_cos_k1_8": float(cos.mean()),
                "progress_cos_by_k": cos.mean(axis=(0, 1)).round(3).tolist(), "path_eff_1_12": float(eff.mean()),
                "frac_frames_err_increases_after_k8": float((e[:, :, 12] > e[:, :, 8] + 0.005).mean())})
    print(f"monotone error decrease (k=1..12): {100 * mono:.1f}% of steps | progress cosine k=1..8: {cos.mean():.3f} "
          f"(by k: {out['progress_cos_by_k']}) | path efficiency 1->12: {eff.mean():.3f} | "
          f"err rises >0.005 from k=8 to 12: {100 * out['frac_frames_err_increases_after_k8']:.1f}%")
    json.dump(out, open(path.replace(".npz", "_convergence.json"), "w"), indent=1)
