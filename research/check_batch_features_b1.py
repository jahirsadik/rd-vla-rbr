import sys
import torch
from research.rdvla_tools import Features, eval_obs, iter_rlds_episodes, load_policy, vlm_features, vlm_features_batch
pol = load_policy(sys.argv[1])
_, ep = next(iter_rlds_episodes())
for t in (0, 20, 40):
    o = eval_obs(ep, t)
    a = vlm_features(pol, o, ep["instruction"]); b = vlm_features_batch(pol, [o], ep["instruction"])
    d = lambda x, y: (x.float() - y.float()).abs().max().item()
    print(f"t={t}: replica(B=1) vs eval path max abs diff  h_a {d(a.h_a, b.h_a):.3g}  h_t {d(a.h_t, b.h_t):.3g}  p {d(a.p, b.p):.3g}")
