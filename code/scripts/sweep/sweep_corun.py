#!/usr/bin/env python3
"""sweep_corun.py -- CPU decode and GPU decode running at the SAME time: how much do they slow each other down?

The two solo sweeps (sweep_cpufreq.py, sweep_gpufreq.py) found no cross effect when one workload runs alone. Here both run together,
so the shared DRAM is the only thing they can fight over (the GPU's host thread sits on an A520 core, so no CPU core is shared).

  cpu    llama-bench, CPU build, A720 x8 threads (decode, tg32), looped back to back for as long as the GPU job runs
  gpu    llama-bench, Vulkan build, -ngl 99, ONE host thread on --host-cpu (an A520), decode tg64 x reps, one process
  jobs   solo_cpu (cpu loop alone), solo_gpu (gpu job alone; the CPU side idle), corun (both) at every (A720 MHz, GPU MHz) pair
Frequencies: the A720 policies are pinned to the job's CPU MHz, the host core's policy to --host-mhz, the GPU to the job's GPU MHz.
Only CPU-loop iterations that lie completely inside the GPU job's decode window (start + --guard s .. end) count for the co-run
CPU number; the GPU number is the GPU process' own average over all its reps (the CPU loop is already running when it starts).
Slowdown = co-run ms/token / solo ms/token - 1, at the same frequencies. Estimated traffic = model size x tokens/s (a sum above the
~40 GB/s CPU ceiling + ~48 GB/s GPU ceiling would mean DRAM is the limit; the DRAM peak itself is not known).

  python3 scripts/sweep/sweep_corun.py                       # full: A720 {1200, 2200} x GPU {350, 600, 1000}, 2 trials
  SUDO_PASS=... python3 scripts/sweep/sweep_corun.py         # non-interactive (sudo only pins frequencies here; no perf)
  python3 scripts/sweep/sweep_corun.py --dry-run
  python3 scripts/sweep/sweep_corun.py --corun-phase prefill --gpu-reps 4   # both sides run prefill (pp --pp) instead of decode
Utilisation over each job's measured window (scripts/sweep/util_sampler.py): cpu_util_a720, cpu_util_gpu_host, gpu_busy, and
bus_GBps_a720/_a520 (perf bus_access, CPU-side traffic only; needs perf_event_paranoid <= 0 or SUDO_PASS).
Frequencies are restored on exit, Ctrl-C and SIGTERM. Raw output: runs/corun_<timestamp>/ (sweep.csv, summary.csv, raw.jsonl, meta.json).
"""
import copy
import csv
import json
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_cpufreq as S   # noqa: E402
import util_sampler as U    # noqa: E402

HERE = S.HERE
COLS = ["trial", "job", "cpu_mhz", "gpu_mhz", "cpu_ts", "cpu_ms_per_tok", "cpu_iters", "cpu_overlap_s", "gpu_ts", "gpu_ms_per_tok",
        "gpu_proc_s", "cpu_est_GBps", "gpu_est_GBps", "freq_ok", "freq_ok_post", "temp_c", "temp_c_post",
        "cpu_util_a720", "cpu_util_gpu_host", "gpu_busy", "read_GBps_a720", "bus_GBps_a720", "bus_GBps_a520"]


def llama_phase(a):
    """'pp' or 'tg' for the CPU loop and the GPU job (sweep_npu.py imports these helpers; it has no --corun-phase: decode)."""
    return "pp" if getattr(a, "corun_phase", "decode") == "prefill" else "tg"


class HostPin:
    """The A520 policy that runs the GPU job's host thread: pinned so its DVFS cannot move the GPU result."""
    def __init__(self, cpu):
        self.cpu = cpu
        pol = int(S.rd(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/related_cpus").split()[0])
        self.d = S.CPUFREQ / f"policy{pol}"
        self.orig = (S.rd(self.d / "scaling_min_freq"), S.rd(self.d / "scaling_max_freq"), S.rd(self.d / "scaling_governor"))

    def pin(self, khz):
        S.wr(self.d / "scaling_governor", "performance"); S.wr(self.d / "scaling_min_freq", 800000)
        S.wr(self.d / "scaling_max_freq", khz); S.wr(self.d / "scaling_min_freq", khz)

    def restore(self):
        try:
            S.wr(self.d / "scaling_max_freq", self.orig[1]); S.wr(self.d / "scaling_min_freq", self.orig[0]); S.wr(self.d / "scaling_governor", self.orig[2])
        except RuntimeError as e:
            print("restore failed:", e, file=sys.stderr)


def cpu_loop(a, stop, out):
    """Back-to-back CPU decode runs (tg, a few reps each); out gets (start, end, tokens/s, model_bytes)."""
    while not stop.is_set():
        try:
            s = time.time()
            r = S.run_llama(a.cpu_bench, 0, llama_phase(a), a.cpu_iter_reps, a, perf=False)
            out.append((s, time.time(), r["avg_ts"], r["model_size"]))
        except Exception as e:                      # noqa: BLE001 - reported by the caller
            out.append(("error", str(e)))
            return


def gpu_job(a):
    ga = copy.copy(a)
    ga.cpus, ga.threads, ga.tg, ga.pp = str(a.host_cpu), 1, a.gpu_tg, getattr(a, "gpu_pp", a.pp)
    s = time.time()
    r = S.run_llama(a.gpu_bench, 99, llama_phase(a), a.gpu_reps, ga, perf=False)
    return s, time.time(), r


def med(v):
    return statistics.median(v) if v else None


def run_job(job, a):
    """job = ('solo_cpu'|'solo_gpu'|'corun'). Returns the measured fields."""
    out, stop = [], threading.Event()
    th = threading.Thread(target=cpu_loop, args=(a, stop, out), daemon=True)
    res = {}
    ul = U.UtilLog({"a720": U.cpulist(a.cpus), "gpu_host": [a.host_cpu]}, {}, perf_bin=a.perf, mode=getattr(a, "perf_mode", None),
                   bus_bytes=a.bus_bytes, tmp_dir=getattr(a, "out_dir_path", None)).start()
    w0 = time.time()
    if job == "solo_gpu":
        gs, ge, r = gpu_job(a)
    else:
        th.start()
        if job == "solo_cpu":
            time.sleep(a.solo_s)
        else:
            time.sleep(a.lead)
            gs, ge, r = gpu_job(a)
        stop.set(); th.join(timeout=120)
    w1 = time.time()
    ul.end()
    lo, hi = (gs + a.guard, ge) if job != "solo_cpu" else (w0 + a.guard, w1)
    res.update({k: v for k, v in ul.window(lo, hi).items() if k in COLS})
    its = [x for x in out if x and x[0] != "error"]
    errs = [x for x in out if x and x[0] == "error"]
    if errs:
        raise RuntimeError(errs[0][1])
    if job != "solo_cpu":
        res.update(gpu_ts=r["avg_ts"], gpu_ms_per_tok=1000.0 / r["avg_ts"], gpu_proc_s=ge - gs,
                   gpu_est_GBps=r["model_size"] * r["avg_ts"] / 1e9 if llama_phase(a) == "tg" else None)   # weights per token: decode only
    if job == "solo_cpu":
        use = its[1:] if len(its) > 1 else its          # the first iteration carries model load and warm-up
    elif job == "corun":
        use = [x for x in its if x[0] >= gs + a.guard and x[1] <= ge]
    else:
        use = []
    if use:
        ts = med([x[2] for x in use])
        res.update(cpu_ts=ts, cpu_ms_per_tok=1000.0 / ts, cpu_iters=len(use), cpu_overlap_s=sum(x[1] - x[0] for x in use),
                   cpu_est_GBps=use[0][3] * ts / 1e9 if llama_phase(a) == "tg" else None)
    return res, {"cpu_iterations": [list(x) for x in its], "gpu_window": [gs, ge] if job != "solo_cpu" else None}


def main():
    ap = S.build_parser(__doc__, freqs="1200,2200", freqs_help="A720 MHz list during the jobs (2200 = the common ceiling of the four policies)")
    ap.add_argument("--gpu-freqs", default="350,600,1000", help="GPU MHz list (available: 72 216 350 600 800 1000)")
    ap.add_argument("--gpu-tg", type=int, default=64, help="decode tokens per rep of the GPU job")
    ap.add_argument("--gpu-reps", type=int, default=8, help="reps of the GPU job (one process; decode dominates its run time)")
    ap.add_argument("--cpu-iter-reps", type=int, default=3, help="reps per CPU-loop iteration (tg32)")
    ap.add_argument("--host-cpu", type=int, default=2, help="core of the GPU job's host thread (an A520)")
    ap.add_argument("--host-mhz", type=int, default=1800, help="MHz that core's policy is pinned to")
    ap.add_argument("--lead", type=float, default=5.0, help="seconds the CPU loop runs before the GPU job starts")
    ap.add_argument("--guard", type=float, default=6.0, help="skip the first seconds of the GPU job (its model load) for the CPU window")
    ap.add_argument("--solo-s", type=float, default=40.0, help="seconds of the solo CPU loop")
    ap.add_argument("--corun-phase", default="decode", choices=["decode", "prefill"], help="what both sides run (prefill = pp --pp / --gpu-pp)")
    ap.add_argument("--gpu-pp", type=int, default=128, help="prompt tokens per rep of the GPU job in --corun-phase prefill")
    ap.set_defaults(trials=2)
    a = ap.parse_args()
    cfs = [int(x) for x in a.freqs.split(",")]
    gfs = [int(x) for x in a.gpu_freqs.split(",")]
    jobs = [("solo_cpu", c, None) for c in cfs] + [("solo_gpu", None, g) for g in gfs] + [("corun", c, g) for c in cfs for g in gfs]
    pre = a.corun_phase == "prefill"
    print(f"plan: A720 {cfs} MHz x GPU {gfs} MHz -> {len(jobs)} jobs per trial ({len(cfs)} solo_cpu, {len(gfs)} solo_gpu, {len(cfs) * len(gfs)} corun), "
          f"{a.trials} trials, phase {a.corun_phase}; CPU loop A720 x{a.threads} {'pp%d' % a.pp if pre else 'tg%d' % a.tg} x{a.cpu_iter_reps} reps, "
          f"GPU job host cpu{a.host_cpu} {'pp%d' % a.gpu_pp if pre else 'tg%d' % a.gpu_tg} x{a.gpu_reps} reps")
    for f in cfs:
        if not 800 <= f <= 2200:
            sys.exit(f"{f} MHz is outside 800..2200")
    if a.dry_run:
        return
    if platform.machine() != "aarch64":
        sys.exit("run this on the board")
    for p in (a.model, a.cpu_bench, a.gpu_bench):
        if not Path(p).exists():
            sys.exit(f"missing: {p}")
    if not os.environ.get("SUDO_PASS") and subprocess.run(["sudo", "-n", "true"]).returncode:
        print("sudo password needed (frequency pinning):")
        if subprocess.run(["sudo", "-v"]).returncode:
            sys.exit("sudo failed")

    out_dir = Path(a.out_dir) if a.out_dir else S.REPO / "runs" / time.strftime("corun_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    a.out_dir_path, a.perf_mode = str(out_dir), U.perf_mode(a.perf)
    print("bus_GBps (perf bus_access):", U.perf_note(a.perf_mode), flush=True)
    fq, host = S.Freq(), HostPin(a.host_cpu)
    (out_dir / "meta.json").write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "kernel": platform.release(),
                                                   "model_bytes": Path(a.model).stat().st_size, "temp_start_c": S.temp_c()}, indent=1))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rows, raw = [], open(out_dir / "raw.jsonl", "w")
    fh = open(out_dir / "sweep.csv", "w", newline="")
    wcsv = csv.DictWriter(fh, fieldnames=COLS)
    wcsv.writeheader()
    cpu_list = [int(c) for c in a.cpus.split(",")]
    try:
        host.pin(a.host_mhz * 1000)
        for trial in range(1, a.trials + 1):
            order = jobs[:]
            random.Random(trial).shuffle(order)
            for job, cf, gf in order:
                cmhz = cf or 2200                           # solo_gpu: the A720 side idles at the ceiling
                fq.pin_cpu(cmhz * 1000)
                if gf:
                    fq.pin_gpu(gf * 1000000)
                else:
                    fq.unpin_gpu()
                time.sleep(1.0)
                ok, cur = S.verify_cpu(cmhz * 1000, cpu_list)
                ok = ok and (gf is None or int(S.rd(S.GPU_DEV / "cur_freq")) // 1000000 == gf)
                S.cooldown(a.cool, a.cool_max)
                t0 = S.temp_c()
                print(f"[trial {trial}] {job:8s} cpu={cf} gpu={gf}  freq_ok={ok}  temp={t0:.0f}C", flush=True)
                res, detail = run_job(job, a)
                raw.write(json.dumps({"trial": trial, "job": job, "cpu_mhz": cf, "gpu_mhz": gf, **detail}) + "\n"); raw.flush()
                ok2, cur2 = S.verify_cpu(cmhz * 1000, cpu_list)
                ok2 = ok2 and (gf is None or int(S.rd(S.GPU_DEV / "cur_freq")) // 1000000 == gf)
                row = {"trial": trial, "job": job, "cpu_mhz": cf, "gpu_mhz": gf, **res, "freq_ok": int(ok), "freq_ok_post": int(ok2),
                       "temp_c": round(t0, 1), "temp_c_post": round(S.temp_c(), 1)}
                wcsv.writerow({k: S.fmt(row.get(k)) for k in COLS}); fh.flush(); rows.append(row)
                print("   " + ("  cpu %.2f ms/tok (%d iters)" % (res["cpu_ms_per_tok"], res["cpu_iters"]) if "cpu_ms_per_tok" in res else "  cpu: no iteration inside the window")
                      + ("  gpu %.2f ms/tok (%.0f s)" % (res["gpu_ms_per_tok"], res["gpu_proc_s"]) if "gpu_ms_per_tok" in res else ""), flush=True)
                if not (ok and ok2):
                    print(f"   WARNING: frequency check failed (pre {ok}, post {ok2}); this row carries freq_ok=0")
    except KeyboardInterrupt:
        print("interrupted; restoring frequencies. Partial results are in", out_dir)
    finally:
        fq.restore(); host.restore()
        fh.close(); raw.close()

    good = [r for r in rows if r["freq_ok"] and r["freq_ok_post"]]

    def mean(job, cf, gf, key):
        v = [r[key] for r in good if r["job"] == job and r["cpu_mhz"] == cf and r["gpu_mhz"] == gf and r.get(key) is not None]
        return statistics.mean(v) if v else None
    out_rows = []
    print(f"\nsummary (mean of {a.trials} trials; failed-frequency rows excluded). slowdown = co-run / solo - 1")
    print(f"{'A720':>5s} {'GPU':>5s} | {'cpu solo':>9s} {'cpu corun':>9s} {'slowdown':>8s} | {'gpu solo':>9s} {'gpu corun':>9s} {'slowdown':>8s} | {'est GB/s cpu+gpu (corun)':>24s}")
    for c in cfs:
        for g in gfs:
            cs, cc = mean("solo_cpu", c, None, "cpu_ms_per_tok"), mean("corun", c, g, "cpu_ms_per_tok")
            gs_, gc = mean("solo_gpu", None, g, "gpu_ms_per_tok"), mean("corun", c, g, "gpu_ms_per_tok")
            ec, eg = mean("corun", c, g, "cpu_est_GBps"), mean("corun", c, g, "gpu_est_GBps")
            sd = lambda x, y: (y / x - 1) if x and y else None           # noqa: E731
            out_rows.append({"cpu_mhz": c, "gpu_mhz": g, "cpu_solo_ms": cs, "cpu_corun_ms": cc, "cpu_slowdown": sd(cs, cc),
                             "gpu_solo_ms": gs_, "gpu_corun_ms": gc, "gpu_slowdown": sd(gs_, gc), "est_cpu_GBps": ec, "est_gpu_GBps": eg})
            pct = lambda v: f"{100 * v:+.1f}%" if v is not None else "-"  # noqa: E731
            f2 = lambda v: f"{v:.2f}" if v is not None else "-"            # noqa: E731
            print(f"{c:>5d} {g:>5d} | {f2(cs):>9s} {f2(cc):>9s} {pct(sd(cs, cc)):>8s} | {f2(gs_):>9s} {f2(gc):>9s} {pct(sd(gs_, gc)):>8s} | "
                  f"{(f'{ec:.1f} + {eg:.1f} = {ec + eg:.1f}' if ec and eg else '-'):>24s}")
    with open(out_dir / "summary.csv", "w", newline="") as sf:
        w = csv.DictWriter(sf, fieldnames=list(out_rows[0]) if out_rows else ["cpu_mhz"])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: S.fmt(v) for k, v in r.items()})
    print("\nresults:", out_dir)


if __name__ == "__main__":
    main()
