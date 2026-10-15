"""Runtime magnitudes inside the adapter: ||S_pre||, ||W_S S_0||, ||W_S S_1||, ||W_pre S_pre|| (per sample, Frobenius
over the 8 slots), and the prelude output RMS, on a few demo frames. Usage: python research/branch_norms.py name=ckpt ..."""
import sys
import torch
from research.rdvla_tools import Features, LoopRunner, eval_obs, iter_rlds_episodes, load_policy, vlm_features_batch
eps = []
for ei, ep in iter_rlds_episodes():
    if ei in (0, 50, 100, 150):
        eps.append(ep)
    if len(eps) == 4:
        break
for arg in sys.argv[1:]:
    name, ckpt = arg.split("=", 1)
    pol = load_policy(ckpt)
    f = Features.cat([vlm_features_batch(pol, [eval_obs(ep, t) for t in (10, 40, 70)], ep["instruction"]) for ep in eps])
    r = LoopRunner(pol.m)
    with torch.no_grad():
        ctx = r.prepare(f)
        s0 = r.init_state(f.h_a.shape[0], ctx, torch.Generator(device="cuda").manual_seed(0))
        s1 = r.step(s0, ctx)
        ws0, wp = r.adapter_parts(s0, ctx)
        ws1, _ = r.adapter_parts(s1, ctx)
        n = lambda x: x.float().norm(dim=(1, 2)).mean().item()
        print(f"{name:18s} |S_pre|={n(ctx['pre']):10.1f}  |W_pre S_pre|={n(wp):10.1f}  |S_0|={n(s0):7.1f}  |W_S S_0|={n(ws0):7.1f}"
              f"  |S_1|={n(s1):7.1f}  |W_S S_1|={n(ws1):7.1f}  |action_queries|={pol.m.action_queries.float().norm().item():7.1f}", flush=True)
    del pol
    torch.cuda.empty_cache()
