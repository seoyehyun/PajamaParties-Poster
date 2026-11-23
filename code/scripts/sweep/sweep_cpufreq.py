#!/usr/bin/env python3
"""sweep_cpufreq.py -- scale the A720 CPU frequency; measure GPU and memory performance at every point. (NPU is not part of it.)

One resource is scaled (the A720 cluster); everything else is held fixed. Per CPU frequency point, all four A720 policies
(0, 6, 8, 10 = cpu0,1,6-11) are pinned to the SAME frequency (performance governor, min = max), then:

  cpu    llama-bench on the CPU build (-ngl 0), THREADS threads on the A720 cores; prefill and decode measured separately.
         -> CPU t/s, and the memory traffic of that run from the core PMU (see below).
  gpu    llama-bench on the Vulkan build (-ngl 99), GPU pinned to --gpu-freq, same host threads; run once, not under perf.
         -> GPU t/s (ms/token) as the CPU frequency changes. The core PMU cannot see GPU traffic (the host cores' share is
            noise-level), so est_GBps (= model size x decode t/s) is the GPU-side traffic estimate.
  probe  membw_probe read / write / triad on the same cores -> memory bandwidth reachable at this CPU frequency.

Memory traffic from the core PMU (armv8_pmuv3_0 = A720), cpu workload only. Calibrated with scripts/sweep/membw_calib.sh
(docs/sweep/results/membw_calib_*.csv), then checked against the llama run itself:
  pmu_read_GBps = l2d_cache_refill x 64 B   cache lines filled into L2 (demand + prefetch). PRIMARY. Exact in the calibration and
                                            within ~10% of model size x decode t/s in a llama decode run (38 vs 42 GB/s).
  pmu_bus_GBps  = bus_access x 31.5 B       exact for pure streaming (30.4-31.9 B/event), but ~1.5x model size x t/s in llama decode:
                                            it also counts traffic that does not reach DRAM (L3 hits / data shared between cores,
                                            unverified). Treat it as an upper bound. A "write" figure by subtraction was dropped:
                                            it came out near 25 GB/s for decode, which is not plausible.
  LLC-load-misses (ll_cache_miss_rd) and LLC-loads (ll_cache_rd) are recorded as llc_miss_ratio / llc_miss_Mps. They are NOT a
  bandwidth (they only count demand loads the prefetcher did not cover: 240 .. 14000 B per event depending on the pattern).
Counters are taken differentially (R_HI reps minus R_LO reps), so model load and warm-up cancel.

Needs: python3, gcc (OpenMP), llama-bench builds, the 6.8 perf binary, sudo (perf_event_paranoid=2).
  python3 scripts/sweep/sweep_cpufreq.py                       # full sweep (asks for the sudo password once if needed)
  SUDO_PASS=... python3 scripts/sweep/sweep_cpufreq.py         # non-interactive
  python3 scripts/sweep/sweep_cpufreq.py --dry-run             # print the plan only
  python3 scripts/sweep/sweep_cpufreq.py --freqs 1200,2200 --trials 1 --workloads cpu,probe      # a quick look
Frequencies are restored on exit, on Ctrl-C and on SIGTERM. Raw output: runs/cpufreq_<timestamp>/ (sweep.csv, summary.csv, raw.jsonl, meta.json), git-ignored;
copy the summary to docs/sweep/results/ when it is worth keeping. (runs/ = raw output, docs/*/results/ = what is committed.)
"""
import argparse
import csv
import json
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                         # scripts/sweep/ -> repo root; raw output goes to REPO/runs/
CPUFREQ = Path("/sys/devices/system/cpu/cpufreq")
POLICIES = (0, 6, 8, 10)                       # the four A720 frequency policies
GPU_DEV = Path("/sys/class/devfreq/CIXH5000:00")
PMU = "armv8_pmuv3_0"                          # cpu0-1,6-11 (A720)
EVENTS = ["cpu_cycles", "inst_retired", "stall_backend_mem", "bus_access", "l2d_cache_refill", "ll_cache_miss_rd", "ll_cache_rd"]
COLS = ["trial", "cpu_mhz", "workload", "phase", "threads", "ts", "ts_sd", "ms_per_tok", "est_GBps", "pmu_scope",
        "pmu_read_GBps", "pmu_bus_GBps", "llc_miss_ratio", "llc_miss_Mps", "stall_mem_frac", "ipc",
        "probe_GBps", "freq_ok", "freq_ok_post", "temp_c", "temp_c_post"]


def build_parser(doc=None, freqs="800,1000,1200,1500,1800,2200", freqs_help="A720 MHz list (2200 is the common ceiling of the four policies)"):
    h = Path.home()
    ap = argparse.ArgumentParser(description=(doc or __doc__).split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freqs", default=freqs, help=freqs_help)
    ap.add_argument("--trials", type=int, default=3, help="repetitions of the whole sweep; the point order is shuffled per trial")
    ap.add_argument("--workloads", default="cpu,gpu,probe", help="subset of cpu,gpu,probe")
    ap.add_argument("--model", default=str(h / "PajamaParties/models/Qwen3-1.7B-Q4_K_M.gguf"))
    ap.add_argument("--cpu-bench", default=str(h / "llama-base/build/bin/llama-bench"))
    ap.add_argument("--gpu-bench", default=str(h / "llama-base/build-vulkan/bin/llama-bench"))
    ap.add_argument("--cpus", default="0,1,6,7,8,9,10,11", help="A720 cores")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pp", type=int, default=128, help="prompt tokens of the prefill test")
    ap.add_argument("--tg", type=int, default=32, help="generated tokens of the decode test")
    ap.add_argument("--r-lo", type=int, default=1, help="repetitions of the short run (counter baseline)")
    ap.add_argument("--r-hi", type=int, default=3, help="repetitions of the long run (reported t/s)")
    ap.add_argument("--gpu-freq", type=int, default=1000, help="GPU MHz pinned during the gpu runs")
    ap.add_argument("--probe-patterns", default="read,write,triad")
    ap.add_argument("--probe-mib", type=int, default=512)
    ap.add_argument("--probe-iters", type=int, default=20)
    ap.add_argument("--bus-bytes", type=float, default=31.5, help="bytes per bus_access event (calibrated)")
    ap.add_argument("--no-pmu", action="store_true", help="run llama-bench without perf (no sudo needed for it)")
    ap.add_argument("--perf", default="/usr/lib/linux-tools/6.8.0-147-generic/perf")
    ap.add_argument("--cool", type=float, default=50.0, help="wait until temp_max <= this (C) before a point")
    ap.add_argument("--cool-max", type=float, default=120.0, help="but at most this many seconds")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def parse_args():
    return build_parser().parse_args()


# ------------------------------------------------------------------ small helpers
def rd(path):
    return Path(path).read_text().strip()


def wr(path, val):                              # sudoers allows `tee` on these sysfs files without a password
    r = subprocess.run(["sudo", "-n", "tee", str(path)], input=f"{val}\n", capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"cannot write {path}: {r.stderr.strip()}")


def perf_direct():
    """True when perf may count system-wide without root (kernel.perf_event_paranoid <= 0)."""
    try:
        return int(rd("/proc/sys/kernel/perf_event_paranoid")) <= 0
    except (OSError, ValueError):
        return False


def sudo_run(cmd):
    if Path(cmd[0]).name == "perf" and perf_direct():   # no password needed for perf then
        return subprocess.run(cmd, capture_output=True, text=True)
    pw = os.environ.get("SUDO_PASS")
    pre = ["sudo", "-S", "-p", ""] if pw else ["sudo", "-n"]
    return subprocess.run(pre + cmd, input=(pw + "\n") if pw else None, capture_output=True, text=True)


def temp_c():
    zs = Path("/sys/class/thermal").glob("thermal_zone*/temp")
    return max(int(rd(z)) for z in zs) / 1000.0


def cooldown(limit, max_s):
    t0 = time.time()
    while temp_c() > limit and time.time() - t0 < max_s:
        time.sleep(2)


class Freq:
    """Pins the A720 policies and the GPU, and puts everything back."""
    def __init__(self):
        self.cpu = {p: (rd(CPUFREQ / f"policy{p}/scaling_min_freq"), rd(CPUFREQ / f"policy{p}/scaling_max_freq"),
                        rd(CPUFREQ / f"policy{p}/scaling_governor")) for p in POLICIES}
        self.gpu = (rd(GPU_DEV / "min_freq"), rd(GPU_DEV / "max_freq"))

    def pin_cpu(self, khz):
        for p in POLICIES:
            d = CPUFREQ / f"policy{p}"
            wr(d / "scaling_governor", "performance"); wr(d / "scaling_min_freq", 800000)
            wr(d / "scaling_max_freq", khz); wr(d / "scaling_min_freq", khz)

    def pin_gpu(self, hz):
        wr(GPU_DEV / "min_freq", self.gpu[0]); wr(GPU_DEV / "max_freq", hz); wr(GPU_DEV / "min_freq", hz)

    def unpin_gpu(self):
        wr(GPU_DEV / "min_freq", self.gpu[0]); wr(GPU_DEV / "max_freq", self.gpu[1])

    def restore(self):
        for p, (mn, mx, gov) in self.cpu.items():
            d = CPUFREQ / f"policy{p}"
            try:
                wr(d / "scaling_max_freq", mx); wr(d / "scaling_min_freq", mn); wr(d / "scaling_governor", gov)
            except RuntimeError as e:
                print("restore failed:", e, file=sys.stderr)
        try:
            self.unpin_gpu()
        except RuntimeError as e:
            print("restore failed:", e, file=sys.stderr)


def verify_cpu(khz, cpus):
    """Load every A720 core for 0.5 s, then read cpuinfo_cur_freq of each policy.
    NOTE: on this board cpuinfo_cur_freq only echoes the requested value (see docs/sweep/results/freq_steps_20261110.csv), so this is NOT proof of
    the real core frequency; the hardware rounds a request up to its next step. Real frequencies need acpi_cppc/feedback_ctrs (scripts/sweep/freq_steps.sh)."""
    procs = [subprocess.Popen(["taskset", "-c", str(c), "timeout", "0.5", "sh", "-c", "while :; do :; done"]) for c in cpus]
    time.sleep(0.3)
    cur = {p: int(rd(CPUFREQ / f"policy{p}/cpuinfo_cur_freq")) for p in POLICIES}
    for p in procs:
        p.wait()
    return all(abs(v - khz) <= 0.03 * khz for v in cur.values()), cur


# ------------------------------------------------------------------ one measurement
def run_llama(binary, ngl, phase, reps, a, perf=True):
    cmd = ["taskset", "-c", a.cpus, binary, "-m", a.model, "-ngl", str(ngl), "-t", str(a.threads),
           "-p", str(a.pp if phase == "pp" else 0), "-n", str(a.tg if phase == "tg" else 0), "-r", str(reps), "-o", "csv"]
    if a.no_pmu or not perf:
        r = subprocess.run(cmd, capture_output=True, text=True)
    else:
        r = sudo_run([a.perf, "stat", "-x,", "-e", ",".join(f"{PMU}/{e}/" for e in EVENTS), "--"] + cmd)
    if r.returncode:
        raise RuntimeError(f"llama-bench failed ({r.returncode}): {r.stderr[-400:]}")
    lines = r.stdout.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"build_commit"') or l.startswith("build_commit"))
    rows = list(csv.DictReader(lines[start:]))
    want_pp = phase == "pp"
    row = next(x for x in rows if (int(x["n_prompt"]) > 0 and int(x["n_gen"]) == 0) == want_pp and (int(x["n_prompt"]) > 0 or int(x["n_gen"]) > 0))
    cnt = {}
    for l in r.stderr.splitlines():
        p = l.split(",")
        if len(p) >= 3 and p[0].strip().isdigit() and f"{PMU}/" in p[2]:
            cnt[p[2].split("/")[1]] = int(p[0])
    return {"avg_ts": float(row["avg_ts"]), "sd_ts": float(row["stddev_ts"]), "avg_ns": float(row["avg_ns"]),
            "model_size": float(row["model_size"]), "counters": cnt, "csv": row}


def measure(binary, ngl, phase, a, raw, tag, pmu=True):
    """pmu=True: short + long run under perf, counters differenced. pmu=False (gpu): one run, no perf."""
    lo = run_llama(binary, ngl, phase, a.r_lo, a) if pmu else None
    hi = run_llama(binary, ngl, phase, a.r_hi, a, perf=pmu)
    raw.write(json.dumps({"tag": tag, "phase": phase, "lo": lo, "hi": hi}) + "\n"); raw.flush()
    out = {"ts": hi["avg_ts"], "ts_sd": hi["sd_ts"], "ms_per_tok": 1000.0 / hi["avg_ts"]}
    if phase == "tg":
        out["est_GBps"] = hi["model_size"] * hi["avg_ts"] / 1e9
    if pmu and not a.no_pmu and set(EVENTS) <= set(lo["counters"]) and set(EVENTS) <= set(hi["counters"]):
        d = {e: hi["counters"][e] - lo["counters"][e] for e in EVENTS}
        dt = (a.r_hi - a.r_lo) * hi["avg_ns"] / 1e9
        if dt > 0 and d["cpu_cycles"] > 0:
            out.update(pmu_read_GBps=d["l2d_cache_refill"] * 64 / dt / 1e9, pmu_bus_GBps=d["bus_access"] * a.bus_bytes / dt / 1e9,
                       llc_miss_Mps=d["ll_cache_miss_rd"] / dt / 1e6, stall_mem_frac=d["stall_backend_mem"] / d["cpu_cycles"],
                       ipc=d["inst_retired"] / d["cpu_cycles"])
            if d["ll_cache_rd"] > 0:
                out["llc_miss_ratio"] = d["ll_cache_miss_rd"] / d["ll_cache_rd"]
    return out


def run_probe(probe, pattern, a):
    r = subprocess.run(["taskset", "-c", a.cpus, "env", "OMP_PROC_BIND=true", str(probe), pattern, str(a.threads),
                        str(a.probe_iters), str(a.probe_mib)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"membw_probe failed: {r.stderr[-300:]}")
    _, _, iters, loop_s, bpi = r.stdout.strip().splitlines()[-1].split(",")
    return int(iters) * float(bpi) / float(loop_s) / 1e9


# ------------------------------------------------------------------ main
def fmt(v):
    return "" if v is None else (f"{v:.4g}" if isinstance(v, float) else v)


def main():
    a = parse_args()
    freqs = [int(x) for x in a.freqs.split(",")]
    wls = a.workloads.split(",")
    pats = a.probe_patterns.split(",")
    per_point = (4 if "cpu" in wls else 0) + (2 if "gpu" in wls else 0) + (len(pats) if "probe" in wls else 0)
    print(f"plan: {len(freqs)} CPU points x {a.trials} trials, {per_point} benchmark runs per point "
          f"= {len(freqs) * a.trials * per_point} runs; workloads={wls}; A720 cores {a.cpus} x{a.threads} threads; model {Path(a.model).name}")
    print(f"      prefill pp{a.pp}, decode tg{a.tg}, reps {a.r_lo}/{a.r_hi}; GPU pinned to {a.gpu_freq} MHz during gpu runs")
    for f in freqs:
        if not 800 <= f <= 2200:
            sys.exit(f"{f} MHz is outside 800..2200 (2200 = highest value all four A720 policies share)")
    if a.dry_run:
        return
    if platform.machine() != "aarch64":
        sys.exit("run this on the board")
    for p in (a.model,) + ((a.cpu_bench,) if "cpu" in wls else ()) + ((a.gpu_bench,) if "gpu" in wls else ()) + (() if a.no_pmu else (a.perf,)):
        if not Path(p).exists():
            sys.exit(f"missing: {p}")
    if not os.environ.get("SUDO_PASS") and subprocess.run(["sudo", "-n", "true"]).returncode:
        print("sudo password needed (perf and the frequency pinning):")
        if subprocess.run(["sudo", "-v"]).returncode:
            sys.exit("sudo failed")

    out_dir = Path(a.out_dir) if a.out_dir else REPO / "runs" / time.strftime("cpufreq_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = out_dir / "membw_probe"
    if "probe" in wls:
        subprocess.run(["gcc", "-O3", "-ffast-math", "-fopenmp", "-o", str(probe), str(HERE / "membw_probe.c")], check=True)

    fq = Freq()
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "kernel": platform.release(), "events": EVENTS,
            "orig_cpu_freq": {str(k): v for k, v in fq.cpu.items()}, "orig_gpu_freq": fq.gpu,
            "model_bytes": Path(a.model).stat().st_size, "temp_start_c": temp_c()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rows = []
    raw = open(out_dir / "raw.jsonl", "w")
    fh = open(out_dir / "sweep.csv", "w", newline="")
    wcsv = csv.DictWriter(fh, fieldnames=COLS)
    wcsv.writeheader()
    cpu_list = [int(c) for c in a.cpus.split(",")]
    try:
        for trial in range(1, a.trials + 1):
            order = freqs[:]
            random.Random(trial).shuffle(order)             # fixed seed per trial: the order is reproducible
            for mhz in order:
                fq.pin_cpu(mhz * 1000)
                ok, cur = verify_cpu(mhz * 1000, cpu_list)
                cooldown(a.cool, a.cool_max)
                t0 = temp_c()
                base = {"trial": trial, "cpu_mhz": mhz, "threads": a.threads, "freq_ok": int(ok), "temp_c": round(t0, 1)}
                print(f"[trial {trial}] cpu {mhz} MHz  freq_ok={ok} {cur}  temp={t0:.0f}C", flush=True)
                point = []
                for wl in [w for w in ("cpu", "gpu") if w in wls]:
                    if wl == "gpu":
                        fq.pin_gpu(a.gpu_freq * 1000000)
                    for phase in ("pp", "tg"):
                        m = measure(a.cpu_bench if wl == "cpu" else a.gpu_bench, 0 if wl == "cpu" else 99, phase, a, raw,
                                    f"t{trial}_{mhz}_{wl}", pmu=(wl == "cpu"))
                        row = {**base, "workload": wl, "phase": phase, "pmu_scope": "cores" if wl == "cpu" else "none", **m}
                        point.append(row)
                        print(f"   {wl:3s} {phase}: {m['ts']:.2f} t/s ({m['ms_per_tok']:.1f} ms/tok)"
                              + (f"  est {m['est_GBps']:.1f} GB/s" if "est_GBps" in m else "")
                              + (f"  mem read {m['pmu_read_GBps']:.1f} GB/s (bus upper bound {m['pmu_bus_GBps']:.1f})  LLC-miss ratio {fmt(m.get('llc_miss_ratio'))}"
                                 if "pmu_read_GBps" in m else ""), flush=True)
                    if wl == "gpu":
                        fq.unpin_gpu()
                if "probe" in wls:
                    for pat in pats:
                        g = run_probe(probe, pat, a)
                        point.append({**base, "workload": "probe", "phase": pat, "probe_GBps": g})
                        print(f"   probe {pat}: {g:.1f} GB/s", flush=True)
                ok2, cur2 = verify_cpu(mhz * 1000, cpu_list)
                t1 = temp_c()
                for row in point:
                    row["freq_ok_post"] = int(ok2); row["temp_c_post"] = round(t1, 1)
                    wcsv.writerow({k: fmt(row.get(k)) for k in COLS})
                fh.flush()
                rows += point
                if not (ok and ok2):
                    print(f"   WARNING: frequency check failed (pre {ok}, post {ok2}: {cur2}); thermal throttling? These rows carry freq_ok=0")
    except KeyboardInterrupt:
        print("interrupted; restoring frequencies. Partial results are in", out_dir)
    finally:
        fq.restore()
        fh.close(); raw.close()

    # summary: mean over trials (rows with a failed frequency check excluded)
    good = [r for r in rows if r.get("freq_ok") and r.get("freq_ok_post")]
    groups = {}
    for r in good:
        groups.setdefault((r["workload"], r["phase"], r["cpu_mhz"]), []).append(r)
    num = ["ms_per_tok", "est_GBps", "pmu_read_GBps", "pmu_bus_GBps", "llc_miss_ratio", "stall_mem_frac", "probe_GBps"]
    with open(out_dir / "summary.csv", "w", newline="") as sf:
        w = csv.writer(sf)
        w.writerow(["workload", "phase", "cpu_mhz", "n"] + num)
        for key in sorted(groups):
            rs = groups[key]
            w.writerow(list(key) + [len(rs)] + [fmt(statistics.mean([r[c] for r in rs if r.get(c) is not None])) if any(r.get(c) is not None for r in rs) else "" for c in num])
    print(f"\nsummary (mean of {a.trials} trials; failed-frequency rows excluded)")
    print(f"{'workload':8s} {'phase':5s} {'MHz':>5s} {'ms/tok':>8s} {'est GB/s':>8s} {'pmu rd':>8s} {'pmu bus':>8s} {'LLCmiss':>8s} {'stallmem':>8s} {'probe':>8s}")
    for key in sorted(groups):
        rs = groups[key]
        mean = lambda c: statistics.mean([r[c] for r in rs if r.get(c) is not None]) if any(r.get(c) is not None for r in rs) else None
        print(f"{key[0]:8s} {key[1]:5s} {key[2]:>5d} " + " ".join(f"{fmt(mean(c)) if mean(c) is not None else '':>8s}" for c in num))
    print("\nresults:", out_dir)


if __name__ == "__main__":
    main()
