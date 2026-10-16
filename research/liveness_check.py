"""Quick liveness check of checkpoints: head-weight distance to the released Spatial head (same seed-42 init),
prelude vs state branch magnitudes in the adapter, and |a_1 - a_32| on 24 demo frames (4 demos x 6 steps).
Usage: python research/liveness_check.py [--suite libero_spatial|libero_object|libero_goal|libero_10] name=ckpt [...]
(the reference head and the demo frames follow --suite; default libero_spatial)"""
import glob, sys
import torch
from research.rdvla_tools import Features, LoopRunner, eval_obs, iter_rlds_episodes, load_policy, vlm_features_batch

def head_sd(d):
    return torch.load(sorted(glob.glob(f"{d}/action_head--*.pt"))[0], map_location="cpu", weights_only=True)
RELEASED = {"libero_spatial": "Spatial_40k", "libero_object": "Object_40k", "libero_goal": "Goal_60k",
            "libero_10": "Long_75k"}
args = sys.argv[1:]
SUITE = "libero_spatial"
if args and args[0] == "--suite":
    SUITE, args = args[1], args[2:]
REF = head_sd(f"outputs/12_24-24_24_{RELEASED[SUITE]}")
mats = [k for k in REF if REF[k].dim() == 2 and "output_proj" not in k]
eps = [ep for ei, ep in iter_rlds_episodes() if ei in (0, 50, 100, 150) or None][:4] if False else []
for ei, ep in iter_rlds_episodes(f"{SUITE}_no_noops"):
    if ei in (0, 50, 100, 150):
        eps.append(ep)
    if len(eps) == 4:
        break
for arg in args:
    name, ckpt = arg.split("=", 1)
    sd = head_sd(ckpt)
    rel = (sum((sd[k].float() - REF[k].float()).pow(2).sum() for k in mats).sqrt() /
           sum(REF[k].float().pow(2).sum() for k in mats).sqrt()).item()
    same = sum((sd[k] == REF[k]).sum().item() for k in mats) / sum(REF[k].numel() for k in mats)
    pol = load_policy(ckpt, SUITE)
    f = Features.cat([vlm_features_batch(pol, [eval_obs(ep, t) for t in (5, 20, 35, 50, 65, 80)], ep["instruction"]) for ep in eps])
    r = LoopRunner(pol.m)
    with torch.no_grad():
        ctx = r.prepare(f)
        g = torch.Generator(device="cuda").manual_seed(0)
        tr = r.trajectory(f, 32, generator=g)
        s0 = r.init_state(f.h_a.shape[0], ctx, torch.Generator(device="cuda").manual_seed(1))
        ws, wp = r.adapter_parts(r.step(s0, ctx), ctx)
    A = tr["actions"]
    n = lambda x: x.float().norm(dim=(1, 2)).mean().item()
    print(f"{name:22s} head vs released: rel diff {rel:.3f}, identical {100*same:5.1f}% | |S_pre| {n(ctx['pre']):8.1f} "
          f"|W_pre S_pre| {n(wp):8.1f} |W_S S_1| {n(ws):6.1f} | |a1-a32| {(A[:,1]-A[:,32]).abs().mean().item():.4f} "
          f"|a4-a32| {(A[:,4]-A[:,32]).abs().mean().item():.4f}", flush=True)
    del pol; torch.cuda.empty_cache()
