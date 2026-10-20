"""Table II for our four retrained checkpoints next to the released checkpoints and the paper.

Our results live under one <src> per suite (full_spatial, full_object, full_goal, full_long). Conventions match the
released Table II in repro/RESULTS.md: Fixed-12 and Binary 5e-4 are 3-seed means (seeds 7/8/9), every other row is
seed 7; k-bar is the mean number of loops per call (all calls of the suite), averaged over the 4 suites.

Usage: python research/table2_compare.py [--out research/out/table2_ours.md]
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repro", "scripts"))
from aggregate import PAPER, PAPER_KBAR, STRAT_NAME, STRAT_ORDER, parse_tag  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
OURS = {"libero_spatial": "full_spatial", "libero_object": "full_object", "libero_goal": "full_goal", "libero_10": "full_long"}
THREE_SEED = {"fixed12", "binary_5e-4"}


def cell(src, tag, suite):
    """(success %, loops per call or None) for one tag/suite, or None if incomplete."""
    seeds = (7, 8, 9) if tag in THREE_SEED else (7,)
    srs, it, calls = [], 0.0, 0
    for s in seeds:
        files = [f"repro/results/{src}/{tag}/{suite}/s{s}/t{t}.json" for t in range(10)]
        if not all(os.path.exists(f) for f in files):
            if s == 7:
                return None
            continue
        succ = eps = 0
        for f in files:
            for e in list(json.load(open(f))["tasks"].values())[0]:
                succ += e["success"]
                eps += 1
                if s == 7:
                    ipc = e.get("iters_per_call") or []
                    if ipc and max(ipc) > 0:
                        it += sum(ipc)
                        calls += len(ipc)
                    elif e.get("avg_iters") is not None:
                        it += e["avg_iters"] * e["num_predictions"]
                        calls += e["num_predictions"]
        srs.append(100 * succ / eps)
    return st.mean(srs), (it / calls if calls else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/out/table2_ours.md")
    a = ap.parse_args()
    tags = sorted({os.path.basename(d) for d in glob.glob("repro/results/released/*")},
                  key=lambda t: (STRAT_ORDER.index(parse_tag(t)[0]), parse_tag(t)[2]))
    lines = ["| Strategy | τ / N | k̄ ours (released, paper) | Spatial | Object | Goal | Long | Avg ours | Avg released | Avg paper |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        kind, val, _ = parse_tag(tag)
        ours = [cell(OURS[s], tag, s) for s in SUITES]
        rel = [cell("released", tag, s) for s in SUITES]
        if any(c is None for c in ours):
            lines.append(f"| {STRAT_NAME[kind]} | {val} | – | (incomplete) | | | | | | |")
            continue
        avg_o = st.mean(c[0] for c in ours)
        avg_r = st.mean(c[0] for c in rel) if all(rel) else float("nan")
        paper = PAPER.get(tag)
        avg_p = f"{st.mean(paper):.1f}" if paper else "–"
        if kind == "fixed":
            kbar = str(val)
        else:
            ko = st.mean(c[1] for c in ours)
            kr = st.mean(c[1] for c in rel) if all(rel) and all(c[1] for c in rel) else float("nan")
            kbar = f"{ko:.2f} ({kr:.2f}, {PAPER_KBAR[tag]:.2f})" if tag in PAPER_KBAR else f"{ko:.2f} ({kr:.2f}, –)"
        cells = []
        for i, c in enumerate(ours):
            ref = f" ({paper[i]:.1f})" if paper else ""
            cells.append(f"{c[0]:.1f}{ref}")
        lines.append(f"| {STRAT_NAME[kind]} | {val} | {kbar} | " + " | ".join(cells) + f" | {avg_o:.1f} | {avg_r:.1f} | {avg_p} |")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
