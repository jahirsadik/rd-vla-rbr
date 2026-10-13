#!/usr/bin/env python
"""Aggregate LIBERO eval JSONs written by eval_suite.sh into Table I / Table II style Markdown.

Tree: repro/results/<src>/<tag>/<suite>/s<seed>/t<task>.json

Tag convention (what eval_suite.sh was called with):
  fixed<N>            fixed recurrence, N loops, 5 of 8 actions executed (code default)
  purekl_<tau>        kl_divergence stop, max 32, 5 actions
  binary_<tau>        purekl + --adaptive_exec True (released direction: >4 loops -> 8 actions)
  lindecay_<tau>      purekl + --use_linear_decay_horizon True
  binpaper_<tau>      purekl + adaptive_exec with threshold 8, low 8, high 4 (paper direction)

Usage:
  aggregate.py                      # print all tables for every <src> found
  aggregate.py --out FILE           # also write them to FILE
  aggregate.py --one DIR            # quick summary for one <suite>/s<seed> dir
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_SHORT = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}

# Paper numbers (success %, 4 suites) keyed by tag, from Table I / Table II.
PAPER = {
    "fixed1": (9.0, 12.2, 11.4, 1.0), "fixed2": (38.0, 61.2, 47.6, 15.0), "fixed4": (79.2, 93.0, 89.2, 74.8),
    "fixed8": (93.0, 97.8, 94.2, 85.2), "fixed12": (92.0, 99.0, 96.0, 84.8), "fixed16": (91.6, 99.0, 95.2, 85.2),
    "fixed24": (92.4, 99.2, 94.2, 86.6), "fixed32": (91.4, 98.8, 93.8, 84.4),
    "binary_1e-4": (91.2, 98.2, 96.0, 79.8), "binary_2e-4": (90.6, 97.2, 96.0, 80.8), "binary_5e-4": (88.6, 98.8, 96.8, 85.8),
    "binary_1e-3": (88.6, 97.8, 94.6, 84.8), "binary_5e-3": (83.2, 93.6, 87.4, 61.8), "binary_1e-2": (74.6, 88.6, 81.6, 43.6),
    "lindecay_1e-4": (91.2, 97.2, 96.2, 80.0), "lindecay_2e-4": (91.2, 98.4, 96.6, 82.0), "lindecay_5e-4": (88.8, 97.6, 94.4, 82.0),
    "lindecay_1e-3": (89.0, 98.4, 94.6, 83.8), "lindecay_5e-3": (84.2, 94.4, 90.6, 62.2), "lindecay_1e-2": (76.2, 87.2, 81.6, 42.6),
    "purekl_1e-4": (91.2, 98.2, 93.8, 81.8), "purekl_2e-4": (92.0, 98.6, 95.0, 82.0), "purekl_5e-4": (90.8, 99.4, 93.2, 82.0),
    "purekl_1e-3": (89.8, 98.2, 93.8, 79.6), "purekl_5e-3": (86.2, 91.4, 87.2, 50.8), "purekl_1e-2": (75.8, 82.2, 81.8, 33.8),
}
PAPER_KBAR = {
    "binary_1e-4": 11.04, "binary_2e-4": 9.71, "binary_5e-4": 7.93, "binary_1e-3": 6.61, "binary_5e-3": 4.27, "binary_1e-2": 3.36,
    "lindecay_1e-4": 11.55, "lindecay_2e-4": 9.86, "lindecay_5e-4": 7.90, "lindecay_1e-3": 6.62, "lindecay_5e-3": 4.26, "lindecay_1e-2": 3.36,
    "purekl_1e-4": 10.58, "purekl_2e-4": 9.25, "purekl_5e-4": 7.66, "purekl_1e-3": 6.57, "purekl_5e-3": 4.30, "purekl_1e-2": 3.38,
}
STRAT_NAME = {"fixed": "Fixed", "binary": "Binary", "lindecay": "Linear Decay", "purekl": "Pure KL", "binpaper": "Binary (paper dir.)"}
STRAT_ORDER = ["fixed", "binary", "lindecay", "purekl", "binpaper"]


def parse_tag(tag):
    m = re.match(r"^fixed(\d+)$", tag)
    if m:
        return "fixed", int(m.group(1)), float(m.group(1))
    m = re.match(r"^(binary|lindecay|purekl|binpaper)_(.+)$", tag)
    if m:
        return m.group(1), m.group(2), float(m.group(2))
    return "other", tag, 0.0


def load_seed_dir(d):
    """Return per-seed stats for one <suite>/s<seed> dir, or None if no JSON."""
    eps = succ = preds = 0
    preds_with_iters = 0
    iters_w = 0.0
    tasks = {}
    errors = 0
    for f in sorted(glob.glob(os.path.join(d, "t*.json"))):
        try:
            data = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"warn: cannot read {f}: {e}", file=sys.stderr)
            continue
        for tname, stats in data.get("tasks", {}).items():
            te = len(stats)
            ts = sum(1 for s in stats if s["success"])
            tasks[tname] = (ts, te)
            eps += te
            succ += ts
            for s in stats:
                n = s.get("num_predictions") or 0
                preds += n
                if s.get("avg_iters") is not None:
                    iters_w += s["avg_iters"] * n
                    preds_with_iters += n
    for f in glob.glob(os.path.join(d, "logs", "*.txt")):
        with open(f, errors="ignore") as fh:
            errors += sum(1 for line in fh if "Episode error" in line)
    if eps == 0:
        return None
    elapsed = None
    ri = os.path.join(d, "run_info.txt")
    if os.path.exists(ri):
        for line in open(ri):
            if line.startswith("elapsed_s:"):
                elapsed = int(line.split(":")[1])
    return {
        "episodes": eps,
        "successes": succ,
        "sr": 100.0 * succ / eps,
        "kbar": (iters_w / preds_with_iters) if preds_with_iters else None,
        "calls_per_ep": preds / eps,
        "errors": errors,
        "tasks": tasks,
        "n_tasks": len(tasks),
        "elapsed_s": elapsed,
    }


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def fmt(m, sd, n, nd=1):
    if m is None:
        return "-"
    s = f"{m:.{nd}f}"
    if n > 1:
        s += f" ± {sd:.{nd}f}"
    return s


def collect(root):
    """results[src][tag][suite] = list of per-seed dicts (with 'seed')."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in sorted(glob.glob(os.path.join(root, "*", "*", "*", "s*"))):
        src, tag, suite, sdir = d.split(os.sep)[-4:]
        st = load_seed_dir(d)
        if st is None:
            continue
        st["seed"] = sdir
        results[src][tag][suite].append(st)
    return results


def table_for_src(src, tags):
    lines = []
    lines.append(f"### Checkpoints: {src}\n")
    lines.append("| Strategy | τ / N | k̄ | calls/ep | Spatial | Object | Goal | Long | Avg | Paper Avg | Δ | seeds | episodes | errors |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    items = []
    for tag, suites in tags.items():
        strat, param, pnum = parse_tag(tag)
        items.append((STRAT_ORDER.index(strat) if strat in STRAT_ORDER else 99, pnum, tag, strat, param, suites))
    for _, _, tag, strat, param, suites in sorted(items):
        cells = []
        suite_means = []
        n_seeds_all = []
        kbars, calls, eps, errs = [], [], 0, 0
        for suite in SUITES:
            runs = suites.get(suite, [])
            n = len(runs)
            n_seeds_all.append(n)
            m, sd = mean_std([r["sr"] for r in runs])
            # flag incomplete suites (fewer than 10 tasks in any seed)
            inc = any(r["n_tasks"] < 10 for r in runs)
            cells.append(fmt(m, sd, n) + ("*" if inc else ""))
            if m is not None:
                suite_means.append(m)
            kbars += [r["kbar"] for r in runs]
            calls += [r["calls_per_ep"] for r in runs]
            eps += sum(r["episodes"] for r in runs)
            errs += sum(r["errors"] for r in runs)
        avg = sum(suite_means) / len(suite_means) if len(suite_means) == 4 else None
        paper = PAPER.get(tag)
        pavg = sum(paper) / 4 if paper else None
        delta = (avg - pavg) if (avg is not None and pavg is not None) else None
        km, ksd = mean_std(kbars)
        cm, csd = mean_std(calls)
        kcell = "-" if km is None else f"{km:.2f}"
        if tag in PAPER_KBAR and km is not None:
            kcell += f" (p {PAPER_KBAR[tag]:.2f})"
        if strat == "fixed":
            kcell = str(param)
        seeds_str = "/".join(str(n) for n in n_seeds_all)
        lines.append(
            f"| {STRAT_NAME.get(strat, strat)} | {param} | {kcell} | {'-' if cm is None else f'{cm:.1f}'} | "
            + " | ".join(cells)
            + f" | {'-' if avg is None else f'{avg:.1f}'} | {'-' if pavg is None else f'{pavg:.1f}'} | "
            + f"{'-' if delta is None else f'{delta:+.1f}'} | {seeds_str} | {eps} | {errs} |"
        )
    lines.append("")
    lines.append("Cells: success % (mean ± std over seeds). `*` = incomplete (fewer than 10 tasks). k̄ = prediction-weighted mean loops per call; (p x) = paper's k̄. seeds = per-suite seed count S/O/G/L.")
    lines.append("")
    return "\n".join(lines)


def per_task_table(src, tag, suites):
    lines = [f"#### {src} / {tag}: per-task success (seed-averaged)\n"]
    for suite in SUITES:
        runs = suites.get(suite, [])
        if not runs:
            continue
        agg = defaultdict(list)
        for r in runs:
            for t, (s, e) in r["tasks"].items():
                agg[t].append(100.0 * s / e if e else None)
        lines.append(f"- {SUITE_SHORT[suite]}: " + ", ".join(f"{t[:40]}={mean_std(v)[0]:.0f}" for t, v in sorted(agg.items())))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="repro/results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--one", default=None, help="summarise a single <suite>/s<seed> dir")
    ap.add_argument("--per-task", action="store_true")
    args = ap.parse_args()

    if args.one:
        st = load_seed_dir(args.one)
        if st is None:
            print("no results in", args.one)
            return
        print(
            f"[aggregate] {args.one}: {st['successes']}/{st['episodes']} = {st['sr']:.1f}% | "
            f"k̄={st['kbar'] if st['kbar'] is None else round(st['kbar'], 2)} | calls/ep={st['calls_per_ep']:.1f} | "
            f"tasks={st['n_tasks']} | episode errors={st['errors']} | elapsed={st['elapsed_s']}s"
        )
        for t, (s, e) in st["tasks"].items():
            print(f"    {t}: {s}/{e}")
        return

    results = collect(args.root)
    out = []
    for src in sorted(results):
        out.append(table_for_src(src, results[src]))
        if args.per_task:
            for tag, suites in results[src].items():
                out.append(per_task_table(src, tag, suites))
    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
