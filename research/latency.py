"""Batch-1 latency anatomy of one RD-VLA control call (run on an otherwise idle GPU).

Times, with CUDA events and medians over repeats: CPU preprocessing (resize/crop + processor), the vision
backbone, the LLM pass, the head, and the whole call, for fixed N loops (original head, and with exact K/V
caching) and for adaptive stopping. Also times the head's pieces (prelude, one loop, one decode) in isolation.
Writes research/out/latency/<name>.json.  Usage: python research/latency.py <ckpt> <name>
"""
import json
import os
import pickle
import statistics
import sys
import time

import torch

from experiments.robot.openvla_utils import get_vla_action
from research.rdvla_tools import NUM_TASK_TOKENS, Features, LoopRunner, head_layers, load_policy

ckpt, name = sys.argv[1], sys.argv[2]
REPS, WARM = 50, 10
pol = load_policy(ckpt)
obs0 = pickle.load(open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb"))
obs = {k: obs0[k] for k in ("full_image", "wrist_image", "state")}
task = obs0.get("task_description", "pick up the black bowl between the plate and the ramekin and place it on the plate")
dev = torch.device("cuda")


class Ev:
    def __init__(self):
        self.a, self.b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def ms(self):
        return self.a.elapsed_time(self.b)


ev = {"vision": Ev(), "llm": Ev(), "head": Ev()}
pol.model.vision_backbone.register_forward_pre_hook(lambda m, i: ev["vision"].a.record())
pol.model.vision_backbone.register_forward_hook(lambda m, i, o: ev["vision"].b.record())
pol.model.language_model.register_forward_pre_hook(lambda m, i, kw=None: ev["llm"].a.record(), with_kwargs=False)
pol.model.language_model.register_forward_hook(lambda m, i, o: ev["llm"].b.record())
runner_cached = LoopRunner(pol.m, kv_cache=True)
layers = head_layers(pol.m)


class TimedHead:
    def __init__(self, mode, n=None, kl=None):
        self.mode, self.n, self.kl = mode, n, kl

    def predict_action(self, ahs, proprio=None, proprio_projector=None, **_):
        ev["head"].a.record()
        if self.mode == "orig_fixed":
            out = pol.head.predict_action(ahs, proprio, proprio_projector, num_iter=self.n)
        elif self.mode == "orig_kl":
            out = pol.head.predict_action(ahs, proprio, proprio_projector, convergence_strategy="kl_divergence",
                                          kl_thresh=self.kl, max_iter=32)
        else:  # cached_fixed
            B = ahs.shape[0]
            p = proprio_projector(proprio.reshape(B, -1).to(torch.bfloat16)).unsqueeze(1)
            hs = ahs[:, layers]
            f = Features(hs[:, :, NUM_TASK_TOKENS:], hs[:, :, :NUM_TASK_TOKENS], p, layers)
            out = runner_cached.picard(f, self.n)
        ev["head"].b.record()
        return out


def time_call(head):
    rows = []
    for i in range(WARM + REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        get_vla_action(pol.cfg, pol.model, pol.processor, dict(obs, state=obs["state"].copy()), task,
                       action_head=head, proprio_projector=pol.proprio_projector, use_film=False, use_minivlm=True)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3
        if i >= WARM:
            rows.append({"wall": wall, **{k: e.ms() for k, e in ev.items()}})
    return {k: round(statistics.median(r[k] for r in rows), 2) for k in rows[0]}


result = {"ckpt": ckpt, "gpu": torch.cuda.get_device_name(0), "reps": REPS, "calls": {}}
# CPU preprocessing alone (resize/crop + processor), as in get_vla_action
from experiments.robot.openvla_utils import prepare_images_for_vla  # noqa: E402
pre = []
for i in range(WARM + REPS):
    t0 = time.perf_counter()
    imgs = prepare_images_for_vla([obs["full_image"], obs["wrist_image"]], pol.cfg)
    _ = [pol.processor("x", im) for im in imgs]
    if i >= WARM:
        pre.append((time.perf_counter() - t0) * 1e3)
result["cpu_preprocess_ms"] = round(statistics.median(pre), 2)

for n in (1, 2, 4, 8, 12, 32):
    result["calls"][f"fixed{n}_original"] = time_call(TimedHead("orig_fixed", n=n))
    result["calls"][f"fixed{n}_kvcache"] = time_call(TimedHead("cached_fixed", n=n))
    print(n, result["calls"][f"fixed{n}_original"], result["calls"][f"fixed{n}_kvcache"], flush=True)
for kl in (1e-4, 5e-4, 1e-3):
    result["calls"][f"adaptive_{kl:g}_original"] = time_call(TimedHead("orig_kl", kl=kl))
    print(kl, result["calls"][f"adaptive_{kl:g}_original"], flush=True)

# head pieces in isolation (batch 1), from captured features
from research.rdvla_tools import vlm_features  # noqa: E402
f = vlm_features(pol, dict(obs, state=obs["state"].copy()), task)
pieces = {}
for label, runner in (("kvcache", runner_cached), ("original", LoopRunner(pol.m, kv_cache=False))):
    with torch.no_grad():
        for piece in ("prepare", "step", "decode"):
            ts = []
            ctx = runner.prepare(f)
            s = runner.init_state(1, ctx)
            for i in range(WARM + REPS):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                if piece == "prepare":
                    runner.prepare(f)
                elif piece == "step":
                    runner.step(s, ctx)
                else:
                    runner.decode(s, ctx)
                b.record()
                torch.cuda.synchronize()
                if i >= WARM:
                    ts.append(a.elapsed_time(b))
            pieces[f"{piece}_{label}"] = round(statistics.median(ts), 3)
result["head_pieces_ms"] = pieces
print(pieces)
os.makedirs("research/out/latency", exist_ok=True)
json.dump(result, open(f"research/out/latency/{name}.json", "w"), indent=1)
print(json.dumps(result, indent=1))
