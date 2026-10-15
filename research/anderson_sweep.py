"""Small Anderson-acceleration sweep on demo frames: gap to the 32-loop Picard decode at loop budgets 4, 6, 8, 12."""
import sys
import numpy as np
import torch
from research.rdvla_tools import Features, LoopRunner, eval_obs, iter_rlds_episodes, load_policy, vlm_features

ckpt = sys.argv[1]
pol = load_policy(ckpt)
frames = []
for ei, ep in iter_rlds_episodes():
    if ei % 7:  # spread over the 432 demos
        continue
    T = len(ep["action"])
    for t in (T // 4, T // 2, 3 * T // 4):
        frames.append(vlm_features(pol, eval_obs(ep, t), ep["instruction"]))
    if len(frames) >= 64:
        break
f = Features.cat(frames[:64])
r = LoopRunner(pol.m)
ctx = r.prepare(f)
s0 = r.init_state(64, ctx, torch.Generator(device="cuda").manual_seed(0))
ref = r.picard(f, 32, state0=s0.clone()).float()
budgets = (4, 6, 8, 12)
def gap(out):
    return round((out.float() - ref).abs().mean().item(), 4)
print("picard          ", [gap(r.picard(f, b, state0=s0.clone())) for b in budgets], flush=True)
for m_hist, beta, project, safeguard in [(3, 1.0, True, True), (3, 1.0, False, True), (3, 1.0, True, False),
                                         (2, 1.0, True, True), (5, 1.0, True, True), (3, 0.5, True, True),
                                         (5, 0.5, False, False), (3, 1.2, True, True)]:
    row = [gap(r.anderson(f, b, m_hist=m_hist, beta=beta, state0=s0.clone(), project=project, safeguard=safeguard))
           for b in budgets]
    print(f"anderson m={m_hist} beta={beta} proj={int(project)} safe={int(safeguard)}", row, flush=True)
# delayed Anderson: plain loops first (leave the noise), then Anderson for the rest of the budget
for warm in (2, 4, 6):
    row = []
    for b in budgets:
        if b <= warm:
            row.append(None); continue
        s = s0.clone()
        for _ in range(warm):
            s = r.step(s, ctx)
        row.append(gap(r.anderson(f, b - warm, m_hist=3, beta=1.0, state0=s)))
    print(f"delayed anderson warm={warm}", row, flush=True)
