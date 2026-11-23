#!/usr/bin/env python3
"""sweep_gpufreq.py -- scale the GPU (Mali) frequency; measure CPU and memory performance at every point. (NPU is not part of it.)

Mirror of sweep_cpufreq.py (it imports its helpers): ONE resource is scaled (the GPU); everything else is held fixed (all four
A720 policies pinned to --cpu-freq, performance governor, min = max). Per GPU frequency point, the GPU is pinned for the whole point:

  gpu     llama-bench on the Vulkan build (-ngl 99), prefill and decode -> the GPU's own sensitivity; est_GBps = model size x decode t/s
  cpu     llama-bench on the CPU build (-ngl 0), THREADS threads on the A720 cores -> CPU ms/token and its memory traffic from the
          core PMU (pmu_read_GBps, see sweep_cpufreq.py). The GPU is idle during it, so a flat line is the expected control result.
  probe   membw_probe read / write / triad on the A720 cores -> CPU-side memory bandwidth at this GPU frequency
  clpeak  OpenCL global-memory bandwidth of the GPU (best of the float .. float16 widths) -> GPU-side memory bandwidth

Needs what sweep_cpufreq.py needs, plus clpeak (OpenCL; the ARM platform / Mali-G720 shows up with `clpeak -p 0`).
  python3 scripts/sweep/sweep_gpufreq.py                    # full sweep
  SUDO_PASS=... python3 scripts/sweep/sweep_gpufreq.py      # non-interactive
  python3 scripts/sweep/sweep_gpufreq.py --dry-run
  python3 scripts/sweep/sweep_gpufreq.py --freqs 350,1000 --trials 1 --workloads gpu,clpeak   # a quick look
GPU points: 216 350 600 800 1000 MHz by default (72 MHz is available but ~3x slower than 216 MHz: pass it explicitly).
Frequencies are restored on exit, on Ctrl-C and on SIGTERM. Raw output: runs/gpufreq_<timestamp>/ (git-ignored).
"""
import csv
import json
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_cpufreq as S   # noqa: E402

HERE = S.HERE
COLS = ["trial", "gpu_mhz", "workload", "phase", "threads", "ts", "ts_sd", "ms_per_tok", "est_GBps", "pmu_scope", "pmu_read_GBps",
        "pmu_bus_GBps", "llc_miss_ratio", "llc_miss_Mps", "stall_mem_frac", "ipc", "probe_GBps", "clpeak_GBps", "freq_ok",
        "freq_ok_post", "gpu_cur_mhz", "temp_c", "temp_c_post"]
SUMMARY_COLS = ["ms_per_tok", "est_GBps", "pmu_read_GBps", "pmu_bus_GBps", "llc_miss_ratio", "stall_mem_frac", "probe_GBps", "clpeak_GBps"]


def run_clpeak(a):
    r = subprocess.run(["taskset", "-c", a.clpeak_cpu, a.clpeak, "-p", "0", "-d", "0", "--global-bandwidth"],
                       capture_output=True, text=True, timeout=300)
    vals = [float(m.group(1)) for m in re.finditer(r"float\d*\s*:\s*([\d.]+)", r.stdout)]
    if not vals:
        raise RuntimeError(f"clpeak gave no bandwidth: {r.stdout[-300:]} {r.stderr[-300:]}")
    return max(vals), r.stdout


def gpu_mhz_now():
    return int(S.rd(S.GPU_DEV / "cur_freq")) // 1000000


def main():
    ap = S.build_parser(__doc__, freqs="216,350,600,800,1000", freqs_help="GPU MHz list (available: 72 216 350 600 800 1000)")
    ap.add_argument("--cpu-freq", type=int, default=2200, help="A720 MHz held fixed (2200 = the common ceiling of the four policies)")
    ap.add_argument("--clpeak", default="clpeak")
    ap.add_argument("--clpeak-cpu", default="2", help="core for the clpeak host thread (an A520, so the A720 cores stay quiet)")
    ap.set_defaults(workloads="cpu,gpu,probe,clpeak")
    a = ap.parse_args()
    freqs = [int(x) for x in a.freqs.split(",")]
    wls = a.workloads.split(",")
    pats = a.probe_patterns.split(",")
    per_point = (4 if "cpu" in wls else 0) + (2 if "gpu" in wls else 0) + (len(pats) if "probe" in wls else 0) + (1 if "clpeak" in wls else 0)
    print(f"plan: {len(freqs)} GPU points x {a.trials} trials, {per_point} benchmark runs per point = {len(freqs) * a.trials * per_point} runs; "
          f"workloads={wls}; A720 fixed at {a.cpu_freq} MHz, cores {a.cpus} x{a.threads} threads; model {Path(a.model).name}")
    print(f"      prefill pp{a.pp}, decode tg{a.tg}, reps {a.r_lo}/{a.r_hi}")
    avail = [int(x) // 1000000 for x in S.rd(S.GPU_DEV / "available_frequencies").split()] if S.GPU_DEV.exists() else [72, 216, 350, 600, 800, 1000]
    for f in freqs:
        if f not in avail:
            sys.exit(f"{f} MHz is not an available GPU frequency {avail}")
    if a.dry_run:
        return
    if platform.machine() != "aarch64":
        sys.exit("run this on the board")
    need = [a.model] + ([a.cpu_bench] if "cpu" in wls else []) + ([a.gpu_bench] if "gpu" in wls else []) + ([] if a.no_pmu else [a.perf])
    for p in need:
        if not Path(p).exists():
            sys.exit(f"missing: {p}")
    if "clpeak" in wls and subprocess.run(["which", a.clpeak], capture_output=True).returncode:
        sys.exit("clpeak not found (apt package clpeak)")
    if not os.environ.get("SUDO_PASS") and subprocess.run(["sudo", "-n", "true"]).returncode:
        print("sudo password needed (perf and the frequency pinning):")
        if subprocess.run(["sudo", "-v"]).returncode:
            sys.exit("sudo failed")

    out_dir = Path(a.out_dir) if a.out_dir else S.REPO / "runs" / time.strftime("gpufreq_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = out_dir / "membw_probe"
    if "probe" in wls:
        subprocess.run(["gcc", "-O3", "-ffast-math", "-fopenmp", "-o", str(probe), str(HERE / "membw_probe.c")], check=True)

    fq = S.Freq()
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "kernel": platform.release(), "events": S.EVENTS,
            "orig_cpu_freq": {str(k): v for k, v in fq.cpu.items()}, "orig_gpu_freq": fq.gpu, "gpu_available_mhz": avail,
            "model_bytes": Path(a.model).stat().st_size, "temp_start_c": S.temp_c()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rows = []
    raw = open(out_dir / "raw.jsonl", "w")
    fh = open(out_dir / "sweep.csv", "w", newline="")
    wcsv = csv.DictWriter(fh, fieldnames=COLS)
    wcsv.writeheader()
    cpu_list = [int(c) for c in a.cpus.split(",")]
    try:
        fq.pin_cpu(a.cpu_freq * 1000)
        for trial in range(1, a.trials + 1):
            order = freqs[:]
            random.Random(trial).shuffle(order)
            for mhz in order:
                fq.pin_gpu(mhz * 1000000)
                time.sleep(1.0)
                ok, cur = S.verify_cpu(a.cpu_freq * 1000, cpu_list)
                g0 = gpu_mhz_now()
                ok = ok and g0 == mhz
                S.cooldown(a.cool, a.cool_max)
                t0 = S.temp_c()
                base = {"trial": trial, "gpu_mhz": mhz, "threads": a.threads, "freq_ok": int(ok), "gpu_cur_mhz": g0, "temp_c": round(t0, 1)}
                print(f"[trial {trial}] gpu {mhz} MHz  freq_ok={ok} (gpu cur {g0} MHz, A720 {cur})  temp={t0:.0f}C", flush=True)
                point = []
                for wl in [w for w in ("gpu", "cpu") if w in wls]:
                    for phase in ("pp", "tg"):
                        m = S.measure(a.cpu_bench if wl == "cpu" else a.gpu_bench, 0 if wl == "cpu" else 99, phase, a, raw,
                                      f"t{trial}_{mhz}_{wl}", pmu=(wl == "cpu"))
                        point.append({**base, "workload": wl, "phase": phase, "pmu_scope": "cores" if wl == "cpu" else "none", **m})
                        print(f"   {wl:3s} {phase}: {m['ts']:.2f} t/s ({m['ms_per_tok']:.1f} ms/tok)"
                              + (f"  est {m['est_GBps']:.1f} GB/s" if "est_GBps" in m else "")
                              + (f"  mem read {m['pmu_read_GBps']:.1f} GB/s  LLC-miss ratio {S.fmt(m.get('llc_miss_ratio'))}" if "pmu_read_GBps" in m else ""),
                              flush=True)
                if "probe" in wls:
                    for pat in pats:
                        g = S.run_probe(probe, pat, a)
                        point.append({**base, "workload": "probe", "phase": pat, "probe_GBps": g})
                        print(f"   probe {pat}: {g:.1f} GB/s", flush=True)
                if "clpeak" in wls:
                    g, txt = run_clpeak(a)
                    raw.write(json.dumps({"tag": f"t{trial}_{mhz}_clpeak", "stdout": txt}) + "\n"); raw.flush()
                    point.append({**base, "workload": "clpeak", "phase": "global_bw", "clpeak_GBps": g})
                    print(f"   clpeak global bandwidth: {g:.1f} GB/s", flush=True)
                ok2, cur2 = S.verify_cpu(a.cpu_freq * 1000, cpu_list)
                ok2 = ok2 and gpu_mhz_now() == mhz
                t1 = S.temp_c()
                for row in point:
                    row["freq_ok_post"] = int(ok2); row["temp_c_post"] = round(t1, 1)
                    wcsv.writerow({k: S.fmt(row.get(k)) for k in COLS})
                fh.flush()
                rows += point
                if not (ok and ok2):
                    print(f"   WARNING: frequency check failed (pre {ok}, post {ok2}: gpu {gpu_mhz_now()} MHz, {cur2}); rows carry freq_ok=0")
    except KeyboardInterrupt:
        print("interrupted; restoring frequencies. Partial results are in", out_dir)
    finally:
        fq.restore()
        fh.close(); raw.close()

    good = [r for r in rows if r.get("freq_ok") and r.get("freq_ok_post")]
    groups = {}
    for r in good:
        groups.setdefault((r["workload"], r["phase"], r["gpu_mhz"]), []).append(r)

    def mean(rs, c):
        v = [r[c] for r in rs if r.get(c) is not None]
        return statistics.mean(v) if v else None
    with open(out_dir / "summary.csv", "w", newline="") as sf:
        w = csv.writer(sf)
        w.writerow(["workload", "phase", "gpu_mhz", "n"] + SUMMARY_COLS)
        for key in sorted(groups):
            w.writerow(list(key) + [len(groups[key])] + [S.fmt(mean(groups[key], c)) if mean(groups[key], c) is not None else "" for c in SUMMARY_COLS])
    print(f"\nsummary (mean of {a.trials} trials; failed-frequency rows excluded)")
    print(f"{'workload':8s} {'phase':9s} {'MHz':>5s} {'ms/tok':>8s} {'est GB/s':>8s} {'pmu rd':>8s} {'pmu bus':>8s} {'LLCmiss':>8s} {'stallmem':>8s} {'probe':>8s} {'clpeak':>8s}")
    for key in sorted(groups):
        print(f"{key[0]:8s} {key[1]:9s} {key[2]:>5d} " + " ".join(f"{S.fmt(mean(groups[key], c)) if mean(groups[key], c) is not None else '':>8s}" for c in SUMMARY_COLS))
    print("\nresults:", out_dir)


if __name__ == "__main__":
    main()
