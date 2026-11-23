#!/usr/bin/env python3
"""Summarize KVSwap stage-1 runs.  usage: summarize_stage1.py <results_dir> [--dev sda]

Reads <dir>/*/{meta.txt,err.txt,iostat.log,mpstat.log} written by run_stage1.sh and writes
<dir>/summary.csv (one row per run) and <dir>/summary_mean.csv (mean over repetitions).
iostat/mpstat are averaged over the LAST `decode seconds` samples, i.e. the decode phase,
so prefill writes do not pollute the numbers.
"""
import csv, math, re, sys
from collections import defaultdict
from pathlib import Path

def parse_meta(p):
    d = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1); d[k] = v
    return d

def parse_err(p):
    t = p.read_text(errors="replace")
    r = {}
    m = re.search(r"prompt eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens.*?([\d.]+) tokens per second", t)
    if m: r["prompt_ms"], r["prompt_tokens"], r["prompt_tps"] = float(m[1]), int(m[2]), float(m[3])
    m = re.search(r"common_perf_print:\s+eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second", t)
    if m: r["decode_ms"], r["n_decode"], r["tpot_ms"], r["decode_tps"] = float(m[1]), int(m[2]), float(m[3]), float(m[4])
    m = re.search(r"Major \(requiring I/O\) page faults:\s*(\d+)", t)
    if m: r["majflt"] = int(m[1])
    m = re.search(r"kvswap_dio: steps=\d+ read_MiB=([\d.]+) reads=(\d+) write_MiB=([\d.]+) writes=\d+ evict_MiB=[\d.]+ stall_s=([\d.]+) write_s=([\d.]+) resident_MiB=([\d.]+)", t)
    if m: r["kv_read_mb"], r["kv_reads"], r["kv_write_mb"], r["io_stall_s"], r["kv_write_s"], r["kv_resident_mb"] = float(m[1]), int(m[2]), float(m[3]), float(m[4]), float(m[5]), float(m[6])
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", t)
    if m: r["max_rss_mb"] = int(m[1]) / 1024
    return r

def blocks_iostat(p, dev):
    """yield dict(col -> float) for `dev`, skipping the first (since-boot) report"""
    rows, cols, first = [], None, True
    for line in p.read_text(errors="replace").splitlines():
        f = line.split()
        if not f: continue
        if f[0] == "Device" or f[0].startswith("Device"):
            cols = f
            if rows or not first:
                pass
            continue
        if cols and f[0] == dev and len(f) == len(cols):
            rows.append({c: float(v) for c, v in zip(cols[1:], f[1:])})
    return rows[1:]  # drop since-boot report

def blocks_mpstat(p):
    """per-second %iowait averaged over the CPUs; handles 12h clocks (AM/PM column)"""
    cols, per = None, defaultdict(list)
    for line in p.read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) > 4 and "%iowait" in f:
            cols = [c for c in f if c not in ("AM", "PM")]; continue
        if f and f[0] == "Average:":
            continue
        if cols:
            g = [x for x in f if x not in ("AM", "PM")]
            if len(g) == len(cols) and g[1].isdigit():
                per[g[1]].append(float(g[cols.index("%iowait")]))
    if not per: return []
    n = min(len(v) for v in per.values())
    return [sum(v[i] for v in per.values()) / len(per) for i in range(n)]

def tail_mean(xs, n):
    xs = xs[-n:] if n > 0 else xs
    return sum(xs) / len(xs) if xs else math.nan

def main():
    args = sys.argv[1:]
    dev = "sda"
    if "--dev" in args:
        i = args.index("--dev"); dev = args[i + 1]; del args[i:i + 2]
    root = Path(args[0])
    rows = []
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        if not (d / "meta.txt").exists(): continue
        meta = parse_meta(d / "meta.txt")
        row = {"run": d.name, "tag": re.sub(r"_r\d+$", "", d.name), "mode": meta.get("mode"),
               "mem": meta.get("mem"), "G": meta.get("G"), "M": meta.get("M"),
               "prefetch": meta.get("prefetch"), "io": meta.get("io"), "depth": meta.get("io_depth"), "exit": meta.get("exit")}
        if (d / "err.txt").exists(): row.update(parse_err(d / "err.txt"))
        n = math.ceil(row.get("decode_ms", 0) / 1000)
        if (d / "iostat.log").exists() and n:
            io = blocks_iostat(d / "iostat.log", dev)
            for col, name in [("r/s", "io_r_s"), ("rkB/s", "io_rkB_s"), ("r_await", "io_r_await_ms"), ("%util", "io_util_pct")]:
                row[name] = tail_mean([r[col] for r in io if col in r], n)
        if (d / "mpstat.log").exists() and n:
            row["iowait_pct"] = tail_mean(blocks_mpstat(d / "mpstat.log"), n)
        te = meta.get("temp_end_mC")
        if te and te.isdigit(): row["temp_end_c"] = int(te) / 1000
        rows.append(row)
    cols = ["run", "tag", "mode", "mem", "G", "M", "prefetch", "io", "depth", "exit", "prompt_tps", "decode_tps", "tpot_ms", "n_decode",
            "majflt", "max_rss_mb", "kv_read_mb", "kv_reads", "kv_write_mb", "io_stall_s", "kv_write_s", "kv_resident_mb", "io_r_s", "io_rkB_s", "io_r_await_ms", "io_util_pct", "iowait_pct", "temp_end_c"]
    def fmt(v):
        if isinstance(v, float): return str(int(v)) if v.is_integer() else f"{v:.2f}"
        return "" if v is None else v
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows: w.writerow([fmt(r.get(c)) for c in cols])
    num = cols[cols.index('prompt_tps'):]
    groups = defaultdict(list)
    for r in rows: groups[r["tag"]].append(r)
    with open(root / "summary_mean.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag", "reps"] + num)
        for tag, rs in groups.items():
            vals = []
            for c in num:
                xs = [r[c] for r in rs if isinstance(r.get(c), (int, float)) and not math.isnan(r[c])]
                vals.append(fmt(sum(xs) / len(xs)) if xs else "")
            w.writerow([tag, len(rs)] + vals)
    print(f"{len(rows)} runs -> {root/'summary.csv'}, {root/'summary_mean.csv'}")
    print(open(root / "summary_mean.csv").read())

main()
