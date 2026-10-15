"""Check vlm_features_batch against the batch-1 path on frames of one demo, and time both."""
import sys, time
import torch
from research.rdvla_tools import (Features, LoopRunner, eval_obs, iter_rlds_episodes, load_policy, vlm_features,
                                  vlm_features_batch)
pol = load_policy(sys.argv[1])
_, ep = next(iter_rlds_episodes())
obs = [eval_obs(ep, t) for t in range(0, 64, 4)]  # 16 frames
torch.cuda.synchronize(); t0 = time.time()
f1 = Features.cat([vlm_features(pol, o, ep["instruction"]) for o in obs]); torch.cuda.synchronize(); t1 = time.time()
fb = vlm_features_batch(pol, obs, ep["instruction"]); torch.cuda.synchronize(); t2 = time.time()
rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
print(f"batch-1: {t1 - t0:.2f}s, batched: {t2 - t1:.2f}s for {len(obs)} frames")
print("max rel diff h_a %.2e h_t %.2e p %.2e" % (rel(fb.h_a, f1.h_a), rel(fb.h_t, f1.h_t), rel(fb.p, f1.p)))
r = LoopRunner(pol.m); ctx = r.prepare(f1); s0 = r.init_state(len(obs), ctx, torch.Generator(device="cuda").manual_seed(0))
a1 = r.picard(f1, 12, state0=s0.clone()).float(); ab = r.picard(fb, 12, state0=s0.clone()).float()
print("12-loop action |batched - batch1| mean %.5f max %.5f" % ((ab - a1).abs().mean().item(), (ab - a1).abs().max().item()))
