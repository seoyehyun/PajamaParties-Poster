#!/usr/bin/env python3
"""CPU x GPU resource-interaction (RI) matrix from the 11/13 sweeps, in the form of CRAVE Fig. 7 (EuroSys '25).

usage: ri_cpu_gpu_proxy.py [--out docs/sweep/results/ri_cpu_gpu_20261119]

CRAVE (4.1.1, 4.2): RI[r][r'] = Spearman(f_r, perf_r'), rows = frequency of r, columns = performance of r'.
Here the cell value is the log-log slope d ln(perf) / d ln(f) instead: with 3-6 monotone points Spearman is only a sign
(+-1), it carries no size. Spearman is kept as small print for the sign. Do not compare cell values with the paper's.
Performance here = tokens/s of the CPU (A720, 8 threads) or GPU (Mali, -ngl 99) Qwen3-1.7B run. No utilisation is needed.
  solo   : only one resource runs, X is swept                 (cpufreq_sweep / gpufreq_sweep summaries; CRAVE-style)
  co-run : CPU decode and GPU decode run together, 2x3 grid    (corun summary; decode only; our extension, not in CRAVE)
           taken at each fixed frequency of the other resource, then averaged
A cell whose tokens/s changes by < NOISE (3 %) over the whole sweep is marked '~0' (its Spearman would be chance). The memory row/column of CRAVE is missing: this board has no memory frequency control.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "docs/sweep/results"


def rows(name):
    with open(R / name, newline="") as f:
        return list(csv.DictReader(f))


NOISE = 0.03


def slope(f, tps):
    x, y = np.log(np.array(f, float)), np.log(np.array(tps, float))
    return float(np.polyfit(x, y, 1)[0])


def rank(v):
    v = np.asarray(v, float)
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def cell(f, tps):
    """(spearman, slope); spearman forced to 0 when the whole response is below NOISE"""
    f, t = np.asarray(f, float), np.asarray(tps, float)
    sl = slope(f, t)
    if (t.max() - t.min()) / t.max() < NOISE:
        return 0.0, sl, True
    rf, rt = rank(f), rank(t)
    return float(np.corrcoef(rf, rt)[0, 1]), sl, False


def solo():
    cpu = rows("cpufreq_sweep_20261113_summary.csv")
    gpu = rows("gpufreq_sweep_20261113_summary.csv")
    out = {}
    for ph, key in (("prefill", "pp"), ("decode", "tg")):
        def get(rs, fk, w):
            pts = [(float(r[fk]), 1000 / float(r["ms_per_tok"])) for r in rs if r["workload"] == w and r["phase"] == key]
            return zip(*sorted(pts))
        m = {}   # (X, Y), 0 = CPU, 1 = GPU
        m[0, 0] = cell(*get(cpu, "cpu_mhz", "cpu"))
        m[0, 1] = cell(*get(cpu, "cpu_mhz", "gpu"))
        m[1, 0] = cell(*get(gpu, "gpu_mhz", "cpu"))
        m[1, 1] = cell(*get(gpu, "gpu_mhz", "gpu"))
        out[ph] = m
    return out


def corun():
    """X's frequency vs Y's tokens/s at each fixed value of the other resource's frequency (so the other axis does not
    confound), then averaged. Groups whose response is below NOISE count as 0."""
    d = [{k: float(v) for k, v in r.items()} for r in rows("corun_20261113_summary.csv")]
    key = {0: "cpu_mhz", 1: "gpu_mhz"}
    ms = {0: "cpu_corun_ms", 1: "gpu_corun_ms"}
    out = {}
    for x in (0, 1):
        for y in (0, 1):
            other = key[1 - x]
            vals, sls, flats = [], [], []
            for o in sorted({r[other] for r in d}):
                g = sorted((r[key[x]], 1000 / r[ms[y]]) for r in d if r[other] == o)
                v, sl, fl = cell(*zip(*g))
                vals.append(v), sls.append(sl), flats.append(fl)
            out[x, y] = (float(np.mean(vals)), float(np.mean(sls)), all(flats))
    return out, d


NAMES = ("CPU", "GPU", "NPU")


def heat(ax, m, title):
    """m[(x, y)] = (spearman, slope, flat) for x = frequency resource, y = performance resource (0 CPU, 1 GPU, 2 NPU)."""
    res = sorted({k[0] for k in m})                       # resources present, e.g. [0, 1] or [0, 1, 2]
    rows_ = res[::-1]                                     # top row = last resource, like CRAVE Fig. 7
    n = len(res)
    arr = np.array([[0.0 if m[x, y][2] else m[x, y][1] for y in res] for x in rows_])
    im = ax.imshow(arr, cmap="RdBu", vmin=-1.0, vmax=1.0)
    fs = 17 if n == 2 else 13
    for i, x in enumerate(rows_):
        for j, y in enumerate(res):
            rho, sl, flat = m[x, y]
            col = "white" if abs(arr[i, j]) > 0.6 else "black"
            ax.text(j, i, "~0" if flat else f"{sl:+.2f}", ha="center", va="center", fontsize=fs, color=col)
            if not flat:
                ax.text(j, i + 0.3, f"rho {rho:+.2f}" if n == 3 else f"Spearman {rho:+.2f}", ha="center", va="center", fontsize=7, color=col)
    ax.set_xticks(range(n), [NAMES[y] + "Perf" for y in res])
    ax.set_yticks(range(n), [NAMES[x] + "Freq" for x in rows_])
    ax.set_title(title, fontsize=11)
    return im


def npu_cells(csv_path, phase):
    """Extra NPU cells from sweep_npu.py's sweep.csv (rows with freq_ok=1). Returns (solo, corun) dicts keyed (x, y)."""
    rs = [r for r in csv.DictReader(open(csv_path, newline="")) if r["freq_ok"] == "1"]
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None   # noqa: E731

    def avg_cell(sel, xk, yk, by=None):
        """mean over groups (fixed value of the other axis `by`) of cell(f_x, tps_y); (nan..) if there is no data."""
        sel = [r for r in rs if sel(r) and f(r, xk) is not None and f(r, yk)]
        if not sel:
            return None
        groups = defaultdict(lambda: defaultdict(list))
        for r in sel:
            groups[r[by] if by else ""][f(r, xk)].append(f(r, yk))
        out = []
        for g in groups.values():
            pts = sorted((x, float(np.mean(v))) for x, v in g.items())
            if len(pts) >= 2:
                out.append(cell(*zip(*pts)))
        if not out:
            return None
        return (float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out])), all(o[2] for o in out))

    solo, co = {}, {}
    ph = lambda r: r["phase"] == phase      # noqa: E731
    solo[2, 2] = avg_cell(lambda r: r["job"] == "npu_solo" and ph(r), "npu_mhz", "npu_ts")
    solo[0, 2] = avg_cell(lambda r: r["job"] == "npu_cpu_solo" and ph(r), "cpu_mhz", "npu_ts")
    solo[1, 2] = avg_cell(lambda r: r["job"] == "npu_gpu_solo" and ph(r), "gpu_mhz", "npu_ts")
    solo[2, 0] = solo[2, 1] = (0.0, 0.0, True)     # NPU frequency while the NPU idles cannot move CPU/GPU (set, not measured)
    co[0, 2] = avg_cell(lambda r: r["job"] == "corun_npu_cpu" and ph(r), "cpu_mhz", "npu_ts", "npu_mhz")
    co[1, 2] = avg_cell(lambda r: r["job"] == "corun_npu_gpu" and ph(r), "gpu_mhz", "npu_ts", "npu_mhz")
    co[2, 0] = avg_cell(lambda r: r["job"] == "corun_npu_cpu" and ph(r), "npu_mhz", "cpu_ts", "cpu_mhz")
    co[2, 1] = avg_cell(lambda r: r["job"] == "corun_npu_gpu" and ph(r), "npu_mhz", "gpu_ts", "gpu_mhz")
    co[2, 2] = avg_cell(lambda r: r["job"] in ("corun_npu_cpu", "corun_npu_gpu") and ph(r), "npu_mhz", "npu_ts", "cpu_mhz")
    return solo, co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "ri_cpu_gpu_20261119"))
    ap.add_argument("--npu-csv-decode", help="sweep.csv of sweep_npu.py run with --phase decode: adds the NPU row/column (3x3)")
    ap.add_argument("--npu-csv-prefill", help="same for --phase prefill (adds NPU to the solo prefill panel)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = solo()
    c, d = corun()

    def extend(m, extra, what):
        miss = [k for k, v in extra.items() if v is None]
        if miss:
            print(f"warning: {what}: no NPU data for cells {miss}; panel stays CPU x GPU")
            return m
        return {**m, **extra}
    if a.npu_csv_decode:
        ns, nc = npu_cells(a.npu_csv_decode, "decode")
        s["decode"], c = extend(s["decode"], ns, "solo decode"), extend(c, nc, "co-run decode")
    if a.npu_csv_prefill:
        ns, _ = npu_cells(a.npu_csv_prefill, "prefill")
        s["prefill"] = extend(s["prefill"], ns, "solo prefill")
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4), gridspec_kw={"wspace": 0.7})
    heat(ax[0], s["prefill"], "solo, prefill\n(CRAVE-style: other resource idle)")
    heat(ax[1], s["decode"], "solo, decode\n(CRAVE-style: other resource idle)")
    im = heat(ax[2], c, "co-run, decode\n(our extension: both run)")
    cpus, gpus = sorted({r["cpu_mhz"] for r in d}), sorted({r["gpu_mhz"] for r in d})
    at = lambda cf, gf: next(r for r in d if r["cpu_mhz"] == cf and r["gpu_mhz"] == gf)
    sl = np.array([[at(cf, gf)["gpu_slowdown"] for gf in gpus] for cf in cpus])
    ax[3].imshow(sl, cmap="Reds", vmin=0, vmax=0.8)
    for i, cf in enumerate(cpus):
        for j, gf in enumerate(gpus):
            r = at(cf, gf)
            ax[3].text(j, i, f"GPU {r['gpu_slowdown']:+.0%}\nCPU {r['cpu_slowdown']:+.0%}", ha="center", va="center",
                       fontsize=10, color="white" if r["gpu_slowdown"] > 0.45 else "black")
    ax[3].set_xticks(range(len(gpus)), [f"{int(g)}" for g in gpus])
    ax[3].set_yticks(range(len(cpus)), [f"{int(c)}" for c in cpus])
    ax[3].set_xlabel("GPU MHz")
    ax[3].set_ylabel("A720 MHz", labelpad=8)
    ax[3].set_title("co-run slowdown vs solo, decode\n(shade = GPU slowdown)", fontsize=11)
    fig.colorbar(im, ax=ax[:3], fraction=0.025, pad=0.03).set_label("d ln(perf) / d ln(f)")
    fig.suptitle("CPU x GPU" + (" x NPU" if a.npu_csv_decode or a.npu_csv_prefill else "") + " RI matrix, Orange Pi 6 Plus, Qwen3-1.7B Q4_K_M (CPU/GPU) (no memory axis; values are slopes, not comparable with the paper's Spearman)", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=160, bbox_inches="tight")
    for name, m in (("solo prefill", s["prefill"]), ("solo decode", s["decode"]), ("co-run decode", c)):
        print(name)
        for (x, y), (v, sl, flat) in sorted(m.items()):
            print(f"  f_{'CGN'[x]} -> perf_{'CGN'[y]}: spearman {v:+.2f}{' (flat, forced 0)' if flat else ''}   slope {sl:+.3f}")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
