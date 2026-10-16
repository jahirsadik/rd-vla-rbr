"""Summarize the closed-loop oracle runs (seed 7) against the released fixed-N / adaptive rows.

Depth oracle (oracle_l3_<eps>): success vs mean loops per call (k*), calls per episode.
Shadow (shadow_<theta>): success vs adopted chunks per episode, and the distribution of
Delta = mean|a_fresh - a_queued| over every queried step.

Usage: python research/analyze_oracles.py [--seed 7] [--out research/out/oracles_summary.json]
"""
import argparse
import glob
import json
import os

import numpy as np

RES = "repro/results"
SUITES = {"libero_spatial": "Spatial", "libero_10": "Long"}


def load_eps(src, tag, suite, seed):
    """Return (episodes, n_tasks) from the per-task JSONs of one run."""
    eps, n_tasks = [], 0
    for p in sorted(glob.glob(f"{RES}/{src}/{tag}/{suite}/s{seed}/t*.json")):
        if p.endswith(".shadow.json"):
            continue
        d = json.load(open(p))
        n_tasks += 1
        for v in d["tasks"].values():
            eps.extend(v)
    return eps, n_tasks


def summarize(eps, n_tasks, fixed_n=None):
    if not eps:
        return None
    succ = np.mean([e["success"] for e in eps]) * 100
    calls = np.array([e["num_predictions"] for e in eps], float)
    ipc = [i for e in eps for i in (e.get("iters_per_call") or [])]
    if ipc and max(ipc) > 0:
        k = np.array(ipc, float)
    elif fixed_n is not None:
        k = np.full(int(calls.sum()), float(fixed_n))
    else:
        avg = [e["avg_iters"] for e in eps if e.get("avg_iters") is not None]
        k = np.array(avg, float) if avg else None
    out = {"tasks": n_tasks, "episodes": len(eps), "success": round(float(succ), 1),
           "calls_per_ep": round(float(calls.mean()), 1)}
    if k is not None:
        out["loops_per_call"] = round(float(k.mean()), 2)
        out["loops_per_ep"] = round(float(k.mean() * calls.mean()), 0)
        if ipc and max(ipc) > 0:
            out["k_p10_p50_p90"] = [float(x) for x in np.percentile(k, [10, 50, 90])]
            out["frac_at_cap20"] = round(float((k >= 20).mean()), 3)
    hz = [h for e in eps for h in (e.get("exec_horizons") or [])]
    if hz:
        out["exec_horizon"] = round(float(np.mean(hz)), 2)
    return out


def shadow_stats(src, tag, suite, seed):
    eps = []
    for p in sorted(glob.glob(f"{RES}/{src}/{tag}/{suite}/s{seed}/t*.json.shadow.json")):
        eps.extend(json.load(open(p))["episodes"])
    if not eps:
        return None
    d = np.array([x for e in eps for x in e["deltas"]], float)
    out = {"episodes": len(eps),
           "success": round(float(np.mean([e["success"] for e in eps]) * 100), 1),
           "queries_per_ep": round(float(np.mean([e["queries"] for e in eps])), 1),
           "adopted_per_ep": round(float(np.mean([e["adopted"] for e in eps])), 1),
           "delta_p50_p90_p99": [round(float(x), 4) for x in np.percentile(d, [50, 90, 99])]}
    for th in (0.03, 0.05, 0.1):
        out[f"frac_delta_gt_{th}"] = round(float((d > th).mean()), 3)
    # Delta by position in the executed chunk (1 = first step after adoption)
    pos = np.array([x for e in eps for x in e.get("pos_in_chunk", [])], int)
    if len(pos) == len(d):
        out["delta_median_by_pos"] = {int(q): round(float(np.median(d[pos == q])), 4)
                                      for q in np.unique(pos)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="research/out/oracles_summary.json")
    a = ap.parse_args()
    summary = {}
    for suite, name in SUITES.items():
        rows = {}
        for n in (1, 2, 4, 8, 12, 16, 24, 32):
            r = summarize(*load_eps("released", f"fixed{n}", suite, a.seed), fixed_n=n)
            if r:
                rows[f"fixed{n}"] = r
        for tag in ("binary_5e-4", "binpaper_5e-4", "purekl_5e-4", "lindecay_5e-4"):
            r = summarize(*load_eps("released", tag, suite, a.seed))
            if r:
                rows[tag] = r
        for eps_ in ("0.005", "0.01", "0.02"):
            r = summarize(*load_eps("released_research", f"oracle_l3_{eps_}", suite, a.seed))
            if r:
                rows[f"oracle_l3_{eps_}"] = r
        for th in ("0.03", "0.05", "inf"):
            r = shadow_stats("released_research", f"shadow_{th}", suite, a.seed)
            if r:
                rows[f"shadow_{th}"] = r
        summary[name] = rows
        print(f"== {name} (seed {a.seed})")
        for k, v in rows.items():
            print(f"  {k:18s} {v}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
