"""Temporal warm start of the recurrent state, offline on the feature cache (pre-research-context.md §8.6 D2).

Replays demos at a fixed call schedule (one policy call every h frames: t = 0, h, 2h, ...) and compares, at every call,
the action after k loops for different starting states S_0:
  cold        fresh TruncNormal noise (what RD-VLA does)
  warm        the previous call's state after its own k loops (chained, i.e. what a deployed warm start would do)
  warm_shift  as warm, but the previous plan's slots are shifted by h (slot i <- slot i+h) and the tail is fresh noise
  warm_conv   the previous call's converged state (32 loops): an upper bound for warm
against the reference a_ref = 32 loops from an independent cold draw. Reports the distance |a_k - a_ref| (mean over
the 8 x 7 normalized action entries) and the L1 error to the demo's normalized label chunk. The first call of every
demo has no previous state and is excluded.

Usage: python research/warm_start_offline.py --cache research/out/cache/released_spatial \
         --ckpt outputs/12_24-24_24_Spatial_40k [--demos 128] [--horizons 5 8] [--out research/out/warm_start.json]
"""
import argparse
import json
import os

import numpy as np
import torch

from research.rdvla_tools import Features, LoopRunner, load_policy, normalized_chunk_labels

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--demos", type=int, default=128)
ap.add_argument("--horizons", type=int, nargs="+", default=[5, 8])
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
ap.add_argument("--out", default="research/out/warm_start.json")
args = ap.parse_args()

dev = "cuda"
pol = load_policy(args.ckpt, suite=args.suite)
m = pol.m
r = LoopRunner(m)
stats = pol.model.norm_stats[pol.cfg.unnorm_key]["action"]
meta = json.load(open(os.path.join(args.cache, "meta.json")))
layers = meta["layers"]
n_all = len([f for f in os.listdir(args.cache) if f.endswith("_h.npy")])
demo_ids = np.linspace(0, n_all - 1, min(args.demos, n_all)).round().astype(int)
K_REF = 32
STARTS = ["cold", "warm", "warm_shift", "warm_conv"]


def gen(seed):
    return torch.Generator(device=dev).manual_seed(seed)


def run(s, ctx, k):
    for _ in range(k):
        s = r.step(s, ctx)
    return s


results = {}
for h in args.horizons:
    # load the frames each demo is called at (t = 0, h, 2h, ...) into CPU memory once
    frames, labels = [], []
    for i in demo_ids:
        hx = np.load(os.path.join(args.cache, f"ep_{i:04d}_h.npy"), mmap_mode="r")
        mz = np.load(os.path.join(args.cache, f"ep_{i:04d}_meta.npz"))
        T = hx.shape[0]
        ts = np.arange(0, T - 8 + 1, h)  # chunks fully inside the demo (as in training)
        frames.append((torch.from_numpy(np.ascontiguousarray(hx[ts])).view(torch.bfloat16),
                       torch.from_numpy(mz["p_bits"][ts].copy()).view(torch.bfloat16)))
        labels.append(np.stack([normalized_chunk_labels(mz["action"], int(t), stats) for t in ts]))
    n_calls = [f[0].shape[0] for f in frames]
    C = max(n_calls)
    N = len(demo_ids)
    D = m.cfg.hidden_dim
    chain = {(st, k): torch.zeros(N, 8, D, device=dev, dtype=torch.bfloat16) for st in STARTS[1:] for k in args.ks}
    prev_conv = torch.zeros(N, 8, D, device=dev, dtype=torch.bfloat16)
    dist = {(st, k): [] for st in STARTS for k in args.ks}
    lerr = {(st, k): [] for st in STARTS for k in args.ks}
    ref_err, floor = [], []
    for c in range(C):
        idx = [j for j in range(N) if n_calls[j] > c]
        hh = torch.stack([frames[j][0][c] for j in idx]).to(dev)  # [B, n_layers, 576, D]
        p = torch.stack([frames[j][1][c] for j in idx]).to(dev).unsqueeze(1)  # [B, 1, D]
        f = Features(hh[:, :, 512:], hh[:, :, :512], p, layers)
        lab = torch.from_numpy(np.stack([labels[j][c] for j in idx])).to(dev).float()
        ii = torch.tensor(idx, device=dev)
        B = len(idx)
        with torch.no_grad():
            ctx = r.prepare(f)
            s_ref = run(r.init_state(B, ctx, gen(10_000 + c)), ctx, K_REF)
            a_ref = r.decode(s_ref, ctx).float()
            a_alt = r.decode(run(r.init_state(B, ctx, gen(20_000 + c)), ctx, K_REF), ctx).float()
            s_cold0 = r.init_state(B, ctx, gen(30_000 + c))
            for k in args.ks:
                outs = {"cold": run(s_cold0, ctx, k)}
                if c > 0:
                    prev = {st: chain[(st, k)][ii] for st in ("warm", "warm_shift")}
                    sh = r.init_state(B, ctx, gen(40_000 + c))
                    sh[:, :8 - h] = prev["warm_shift"][:, h:] if h < 8 else sh[:, :0]
                    outs["warm"] = run(prev["warm"], ctx, k)
                    outs["warm_shift"] = run(sh, ctx, k)
                    outs["warm_conv"] = run(prev_conv[ii], ctx, k)
                else:  # first call of every demo: cold start for all chains
                    for st in STARTS[1:]:
                        outs[st] = outs["cold"]
                for st, s in outs.items():
                    if st != "cold":
                        chain[(st, k)][ii] = s if st != "warm_conv" else chain[(st, k)][ii]
                    if c == 0:
                        continue
                    a = r.decode(s, ctx).float()
                    dist[(st, k)].append((a - a_ref).abs().mean(dim=(1, 2)).cpu())
                    lerr[(st, k)].append((a - lab).abs().mean(dim=(1, 2)).cpu())
            prev_conv[ii] = s_ref
            if c > 0:
                ref_err.append((a_ref - lab).abs().mean(dim=(1, 2)).cpu())
                floor.append((a_alt - a_ref).abs().mean(dim=(1, 2)).cpu())
    fl = torch.cat(floor)
    res = {"calls": int(fl.numel()), "demos": N, "noise_floor_median": float(fl.median()),
           "ref_label_l1": float(torch.cat(ref_err).mean()), "rows": {}}
    print(f"\n== h={h}: {res['calls']} calls from {N} demos | S0-noise floor (32 vs 32 loops) median "
          f"{res['noise_floor_median']:.4f} | label L1 of 32-loop action {res['ref_label_l1']:.4f}")
    print(f"{'start':11s} " + " ".join(f"{'k=' + str(k):>22s}" for k in args.ks))
    for st in STARTS:
        row = {}
        cells = []
        for k in args.ks:
            d = torch.cat(dist[(st, k)])
            le = torch.cat(lerr[(st, k)])
            row[k] = {"dist_median": float(d.median()), "dist_mean": float(d.mean()),
                      "frac_le_0.01": float((d <= 0.01).float().mean()), "frac_le_0.02": float((d <= 0.02).float().mean()),
                      "label_l1": float(le.mean())}
            cells.append(f"{row[k]['dist_median']:.4f}/{row[k]['frac_le_0.02']:.2f}/{row[k]['label_l1']:.3f}")
        res["rows"][st] = row
        print(f"{st:11s} " + " ".join(f"{x:>22s}" for x in cells))
    print("(cells: median |a_k - a_ref| / fraction of calls within 0.02 / label L1)")
    results[f"h{h}"] = res
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(results, open(args.out, "w"), indent=1)
