"""Feature cache for head-only experiments (ideas Part 1 §0.5): run one checkpoint's VLM once over every frame of a
LIBERO RLDS dataset and store exactly what the action head reads.

Per demo (episode index i, T steps), in <out>/:
  ep_{i:04d}_h.npy     uint16 [T, n_layers, 576, 896]: bf16 bits of the VLM hidden states at the head's layers
                       (`layers` in meta.json, e.g. [11, 23]); token order = 512 vision patches then 64 latent tokens,
                       i.e. h_t = h[:, :, :512], h_a = h[:, :, 512:]. View as bf16 with torch.from_numpy(x).view(torch.bfloat16).
  ep_{i:04d}_meta.npz  state [T, 8] (raw), proprio_norm [T, 8] (as fed to the projector), p_bits uint16 [T, 896]
                       (bf16 bits of this checkpoint's projected proprio token), action [T, 7] (raw), instruction.
Frames go through the eval preprocessing (resize 224 + center crop, no augmentation) and the batched VLM pass of
`vlm_features_batch` (bf16, batch numerics like training). Training uses steps t <= T - 8 (8-step chunks);
all T steps are stored. Resumable: finished demos are skipped.

Usage: python research/build_feature_cache.py --ckpt <dir> --out research/out/cache/<name> [--dataset ... --suite ...]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import experiments.robot.openvla_utils as OU
from research.rdvla_tools import eval_obs, iter_rlds_episodes, load_policy, vlm_features_batch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default="libero_spatial_no_noops")
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

pol = load_policy(args.ckpt, suite=args.suite)
pstats = pol.model.norm_stats[pol.cfg.unnorm_key]["proprio"]
meta = {"ckpt": args.ckpt, "dataset": args.dataset, "suite": args.suite, "unnorm_key": pol.cfg.unnorm_key,
        "layers": None, "token_order": "512 vision patches, then 64 latent tokens", "dtype": "bf16 bits as uint16",
        "preprocessing": "eval path: resize_image_for_policy(224) + center crop; batched VLM pass"}
t0, n_frames, n_eps = time.time(), 0, 0
for ei, ep in iter_rlds_episodes(args.dataset):
    hp, mp = os.path.join(args.out, f"ep_{ei:04d}_h.npy"), os.path.join(args.out, f"ep_{ei:04d}_meta.npz")
    if os.path.exists(mp):
        continue
    T = len(ep["action"])
    hs, ps = [], []
    for s in range(0, T, args.batch):
        f = vlm_features_batch(pol, [eval_obs(ep, t) for t in range(s, min(T, s + args.batch))], ep["instruction"])
        meta["layers"] = f.layers
        h = torch.cat([f.h_t, f.h_a], dim=2)  # [b, n_layers, 576, D]
        hs.append(h.contiguous().view(torch.int16).cpu().numpy().view(np.uint16))
        ps.append(f.p[:, 0].contiguous().view(torch.int16).cpu().numpy().view(np.uint16))
    np.save(hp + ".tmp.npy", np.concatenate(hs))
    os.replace(hp + ".tmp.npy", hp)
    proprio_norm = np.stack([OU.normalize_proprio(st.copy(), pstats) for st in ep["state"]]).astype(np.float32)
    np.savez(mp, state=ep["state"], proprio_norm=proprio_norm, p_bits=np.concatenate(ps), action=ep["action"],
             instruction=ep["instruction"])
    n_frames += T
    n_eps += 1
    if n_eps % 20 == 0:
        el = time.time() - t0
        print(f"[cache] {ei + 1} demos, {n_frames} new frames, {n_frames / el:.1f} frames/s, {el / 60:.1f} min", flush=True)
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
print(f"[cache] done: {n_eps} new demos, {n_frames} frames in {(time.time() - t0) / 60:.1f} min", flush=True)
