#!/usr/bin/env python3
"""RI analysis for sweep_ri.sh results (standard library only).

usage: ri.py <ri.csv> [<ri.csv> ...]  [--matrix] [--out DIR]

Input is the ri.csv written by scripts/sweep/sweep_ri.sh. Several files may be given (e.g. a CPU sweep and a GPU
sweep, or in-memory and offloading runs); rows are labelled by the 'mode' column.

Per (mode, resource A, frequency) it averages the repetitions and prints:
  1. a table of throughput, relative throughput and the sensitivity of each phase
  2. sensitivity S per phase:  S = (d tps / tps) / (d f / f), a log-log slope between the lowest and the highest
     frequency.  1.0 = throughput scales with the frequency (that resource is the bottleneck), 0 = it does not.
     Spearman rank correlation is also reported, but with a handful of points it is only a monotonicity check.
  3. knee: the lowest frequency whose throughput is >= 95 % of the maximum
  4. if utilisation columns exist (util_sel_p / util_sel_d / iowait_d): how they change with frequency
Not implemented yet: PPR / Pareto (needs the power model) and the dominant-resource utility.

Runs with exit != 0 or freq_ok == 0 are dropped and listed.
The 4x4 (here: resource x resource) RI matrix needs sweeps on more than one resource; each file gives one row of it
(how the sensitivity of the measured workload to resource A looks).  --matrix prints the collected rows together.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def slope(freqs, tps):
    """log-log slope between the lowest and the highest frequency; also the slope of each neighbouring pair"""
    pts = sorted(zip(freqs, tps))
    pts = [(f, t) for f, t in pts if f and t and f > 0 and t > 0]
    if len(pts) < 2:
        return None, []
    (f0, t0), (f1, t1) = pts[0], pts[-1]
    total = math.log(t1 / t0) / math.log(f1 / f0)
    local = [(pts[i][0], pts[i + 1][0], math.log(pts[i + 1][1] / pts[i][1]) / math.log(pts[i + 1][0] / pts[i][0]))
             for i in range(len(pts) - 1)]
    return total, local


def load(paths):
    rows, dropped = [], []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                r["_src"] = os.path.basename(os.path.dirname(os.path.abspath(p)))
                if r.get("exit") not in ("0", "", None):
                    dropped.append((r["_src"], r.get("res_a"), r.get("freq_a_mhz"), r.get("rep"), "exit=" + str(r.get("exit"))))
                    continue
                if r.get("freq_ok") == "0":
                    dropped.append((r["_src"], r.get("res_a"), r.get("freq_a_mhz"), r.get("rep"), "freq_ok=0"))
                    continue
                rows.append(r)
    return rows, dropped


def group(rows):
    """(mode, res_a, res_b, freq_b) -> {freq_a: [rows]}"""
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[(r["mode"], r["res_a"], r.get("res_b") or "", r.get("freq_b_mhz") or "")][num(r["freq_a_mhz"])].append(r)
    return g


def f1(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def analyse(rows, args):
    out = []
    summary = []
    for key, byf in sorted(group(rows).items()):
        mode, ra, rb, fb = key
        title = f"{mode}  sweep of {ra}" + (f"  (with {rb} fixed at {fb} MHz)" if rb else "")
        freqs = sorted(byf)
        print(f"\n=== {title} ===")
        cols = [("pp_tps", "prefill t/s"), ("tg_tps", "decode t/s"), ("tpot_ms", "TPOT ms"), ("temp_max_mC", "Tmax C")]
        util = [c for c in ("util_sel_p", "util_sel_d", "util_all_d", "iowait_d") if any(num(r.get(c)) is not None for r in rows)]
        head = ["f MHz", "n"] + [c[1] for c in cols] + util
        print("  ".join(f"{h:>12}" for h in head))
        agg = {}
        for f in freqs:
            rs = byf[f]
            vals = {c: mean([num(r.get(c)) for r in rs]) for c, _ in cols}
            vals.update({c: mean([num(r.get(c)) for r in rs]) for c in util})
            vals["pp_sd"] = sd([num(r.get("pp_tps")) for r in rs])
            vals["tg_sd"] = sd([num(r.get("tg_tps")) for r in rs])
            vals["n"] = len(rs)
            agg[f] = vals
            line = [f"{f:.0f}", str(len(rs)), f1(vals["pp_tps"]), f1(vals["tg_tps"]), f1(vals["tpot_ms"], 1),
                    f1(vals["temp_max_mC"] / 1000 if vals["temp_max_mC"] else None, 0)] + [f1(vals[c], 1) for c in util]
            print("  ".join(f"{v:>12}" for v in line))
        if len(freqs) < 2:
            print("  (one frequency only: no sensitivity)")
            continue
        for ph, col, label in (("prefill", "pp_tps", "pp"), ("decode", "tg_tps", "tg")):
            t = [agg[f][col] for f in freqs]
            if None in t:
                continue
            s, local = slope(freqs, t)
            rho = spearman(freqs, t)
            top = max(t)
            knee = next((f for f, v in zip(freqs, t) if v >= 0.95 * top), None)
            print(f"  {ph:8s} S = {f1(s, 3)}   spearman = {f1(rho, 2)}   knee(95% of max) = {f1(knee, 0)} MHz")
            print("           local slope per step: " + ", ".join(f"{a:.0f}->{b:.0f}: {l:.2f}" for a, b, l in local))
            summary.append({"mode": mode, "res_a": ra, "res_b": rb, "freq_b": fb, "phase": ph, "S": s,
                            "spearman": rho, "knee_mhz": knee, "f_min": freqs[0], "f_max": freqs[-1],
                            "tps_min": t[0], "tps_max": t[-1]})
        for c in util:
            v = [agg[f][c] for f in freqs]
            if None not in v:
                rho = spearman(freqs, v)
                print(f"  {c:11s} {f1(v[0], 1)} -> {f1(v[-1], 1)} %   spearman with frequency = {f1(rho, 2)}")
    return summary


def matrix(summary):
    print("\n=== sensitivity matrix (rows = swept resource, columns = phase; one measured workload per mode) ===")
    by = defaultdict(dict)
    for s in summary:
        by[(s["mode"], s["res_b"], s["freq_b"])][(s["res_a"], s["phase"])] = s["S"]
    for key, d in by.items():
        ress = sorted({k[0] for k in d})
        print(f"  mode={key[0]}" + (f" (with {key[1]} at {key[2]} MHz)" if key[1] else ""))
        print("  " + f"{'swept':>8}{'prefill S':>12}{'decode S':>12}")
        for r in ress:
            print("  " + f"{r:>8}{f1(d.get((r, 'prefill')), 3):>12}{f1(d.get((r, 'decode')), 3):>12}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--matrix", action="store_true", help="also print the sensitivity matrix over all files")
    ap.add_argument("--out", help="write sensitivity.csv here")
    args = ap.parse_args()
    rows, dropped = load(args.csv)
    if dropped:
        print(f"dropped {len(dropped)} runs:")
        for d in dropped:
            print("   ", *d)
    if not rows:
        sys.exit("no usable rows")
    summary = analyse(rows, args)
    if args.matrix:
        matrix(summary)
    if args.out and summary:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "sensitivity.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
