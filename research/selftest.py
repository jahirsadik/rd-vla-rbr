"""Smoke test for research/rdvla_tools.py on one checkpoint: feature capture, K/V-cache exactness,
Anderson and oracle runs, and demo-frame loading. Usage: python research/selftest.py <ckpt_dir>"""
import pickle
import sys
import time

import numpy as np
import torch

from research.rdvla_tools import (LoopRunner, check_kv_cache_exact, eval_obs, iter_rlds_episodes, load_policy,
                                  normalized_chunk_labels, vlm_features)

ckpt = sys.argv[1]
pol = load_policy(ckpt)
obs = pickle.load(open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb"))
print("sample obs keys:", sorted(obs.keys()))
f = vlm_features(pol, obs, "pick up the black bowl between the plate and the ramekin and place it on the plate")
print("features:", tuple(f.h_a.shape), tuple(f.h_t.shape), tuple(f.p.shape), f.layers, f.h_a.dtype)
print("kv-cache exactness:", check_kv_cache_exact(pol, f, n=12))

r = LoopRunner(pol.m)
tr = r.trajectory(f, K=32, generator=torch.Generator(device="cuda").manual_seed(0))
A = tr["actions"][0]
print("|a_k - a_32| for k=1,2,4,8,12:", [round((A[k] - A[32]).abs().mean().item(), 4) for k in (1, 2, 4, 8, 12)])
print("rho_adapt k=1,2,12:", [round(tr["rho"][0, k - 1].item(), 3) for k in (1, 2, 12)])
out, res = r.anderson(f, budget=8, return_residuals=True)
print("anderson residuals:", [round(x, 4) for x in res[0].tolist()])
a_or, k = r.oracle_l3(f, eps=0.01)
print("oracle k* (eps 0.01):", k.tolist())

stats = pol.model.norm_stats[pol.cfg.unnorm_key]["action"]
t0 = time.time()
for i, ep in iter_rlds_episodes():
    T = len(ep["action"])
    t = T // 2
    lab = normalized_chunk_labels(ep["action"], t, stats)
    fe = vlm_features(pol, eval_obs(ep, t), ep["instruction"])
    a12 = r.picard(fe, 12)[0].float().cpu().numpy()
    print(f"episode {i} '{ep['instruction']}' T={T}: |a12 - label| = {np.abs(a12 - lab).mean():.4f} "
          f"(gripper label {lab[0, 6]:.0f}, pred {a12[0, 6]:.2f}); load {time.time() - t0:.1f}s")
    if i >= 2:
        break
