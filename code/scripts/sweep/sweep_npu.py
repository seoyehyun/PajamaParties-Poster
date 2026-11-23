#!/usr/bin/env python3
"""sweep_npu.py -- fill the NPU row and column of the CPU x GPU x NPU RI matrix. Run on the board. NOT yet run on hardware.

NPU job = a command you supply (--npu-cmd) that runs the NPU LLM (prefill or decode .cix, NOE_Engine) and prints
    NPU_RESULT {"tps": <tokens/s>, "tokens": <n>}
{phase} in the command is replaced by prefill|decode ({mhz} is not needed: this script pins the NPU). The NPU host process is pinned to
--npu-host-cpu (an A520, default cpu3, so it shares no core with the A720 CPU job and not with the GPU host on cpu2).

Two ways to keep the NPU busy during a window:
  default            the command is run back to back (one process per iteration, like the CPU loop in sweep_corun.py). Every iteration pays
                     the engine load, and the NPU idles meanwhile, so a co-run window UNDERESTIMATES the interference.
  --npu-persistent   the command is started ONCE per job and must print one NPU_RESULT line per inference, forever, until it is killed.
                     The NPU stays busy; use this for every co-run. (Check first that the aipu driver survives many inferences in one
                     process: docs/npu/npu-llm-plan.md section 7.)

  jobs (per --phase prefill|decode; the CPU and GPU sides always run decode)
    npu_solo         NPU alone, NPU swept over --npu-freqs, A720 at --cpu-solo-mhz, GPU idle           -> NPUFreq -> NPUPerf
    npu_cpu_solo     NPU alone at its top frequency, A720 swept over --cpu-freqs                       -> CPUFreq -> NPUPerf
    npu_gpu_solo     NPU alone at its top frequency, GPU swept over --gpu-freqs                        -> GPUFreq -> NPUPerf
    cpu_solo / gpu_solo   the CPU or GPU decode alone at each frequency (denominators of the slowdowns; NPU idle)
    corun_npu_cpu    NPU loop + CPU decode loop, NPU x A720 grid
    corun_npu_gpu    NPU loop + GPU decode job,  NPU x GPU grid
    corun_npu_cpu_gpu  all three at once, NPU x A720 x GPU grid (--tri-npu-freqs/--tri-cpu-freqs/--tri-gpu-freqs; --no-tri to skip)
    npu_heat         (--with-heat) FIXED WORK: --heat-iters NPU inferences at each NPU frequency (and the default governor), then the
                     cool-down tail. heat_area_Cs = integral of (T - T_start) dt over work + tail, heat_per_tok = that / tokens. Same
                     energy proxy as docs/joint (a lower frequency finishes later: the area shows whether it still pays off).

Every job also records the temperature trace over its window: temp_peak_c and heat_area_Cs (integral of T - T_start; the max over all thermal
zones, so it is a coarser proxy than the zone-8 area of thermal_ab.py).

Outputs in runs/npu_<timestamp>/: sweep.csv (one row per job run), summary.csv (means over trials plus the slowdowns), raw.jsonl, meta.json.
summary.csv columns: n, npu_ts/cpu_ts/gpu_ts (tokens/s), npu_est_GBps, npu_busy, heat, and for every co-run row
    <x>_slowdown   = solo ms/token ratio: corun_ms / solo_ms - 1            (same convention as the CPU x GPU tables)
    <x>_speed_loss = 1 - corun tokens/s / solo tokens/s                       (the "speed dropped by N %" form)
  with x in npu, cpu, gpu. solo = npu_solo at the same NPU MHz, cpu_solo at the same A720 MHz, gpu_solo at the same GPU MHz.
  For the 3-way rows *_extra_vs_pair = 1 - tri tokens/s / pair tokens/s (what the third resource adds on top of the pair).
Estimated NPU traffic = --npu-model-bytes x tokens/s (decode only; give the weight bytes actually read per token). It is an ESTIMATE:
the A720 PMU cannot see NPU traffic. npu_busy = delta runtime_active_time / wall time of the NPU devfreq device over the window
(runtime PM: it stays ~1.0 while a persistent process holds the device, so it is NOT an occupancy).

Utilisation over the same window (1 Hz samples; the GPU job's window starts after --guard s, like the slowdowns):
  npu_util            sum(npu_ms) / sum(unit wall) of the counted NPU iterations: the share of the loop the NPU is computing, from the
                      NPU_RESULT extras of scripts/npu/llm/npu_llm_run.py (empty for a command that does not print npu_ms)
  npu_host_ms_per_tok host time per token of the NPU loop (input load, output read, argmax, KV shift on the A520)
  npu_ttft_ms         prefill phase only: median wall time of one prefill unit (S=256 tokens) = the TTFT proxy, like prompt eval time
  gpu_busy            delta runtime_active_time / wall of the Mali device (runtime PM; the GPU suspends when idle, so a coarse busy share)
  cpu_util_a720 / cpu_util_npu_host / cpu_util_gpu_host / cpu_util_all   busy share from /proc/stat (A720 job cores, --npu-host-cpu,
                      --host-cpu, all cores)
  disk_util_nvme / disk_util_sda  io_ticks share from /proc/diskstats; disk_rd_MBps / disk_wr_MBps summed over the whole disks
  read_GBps_a720      l2d_cache_refill x 64 B (A720, the primary DRAM read proxy, as pmu_read_GBps of the 11/13 sweeps)
  bus_GBps_a720 / bus_GBps_a520   system-wide bus_access x --bus-bytes per PMU (perf -a -I 1000): an upper bound (~1.7x read in decode). CPU-side traffic only: neither NPU nor
                      GPU traffic is visible to these PMUs and the board has no uncore (DSU/CMN/DMC) PMU. --bus-bytes is calibrated on the
                      A720 only. Needs perf_event_paranoid <= 0 or SUDO_PASS (perf is not in the password-less sudoers list); else empty.

  python3 scripts/sweep/sweep_npu.py --npu-cmd 'python3 ~/npu/run_cix.py --phase {phase}' --npu-persistent --phase decode --dry-run
  python3 scripts/sweep/sweep_npu.py --analyze runs/npu_<ts>/sweep.csv        # rebuild summary.csv from a sweep.csv (no board needed)
Frequencies (A720, GPU, A520 host, NPU governor/set_freq) are restored on exit, Ctrl-C and SIGTERM.
"""
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
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_cpufreq as S      # noqa: E402
import sweep_corun as C        # noqa: E402
import util_sampler as U      # noqa: E402

NPU_DEV = Path("/sys/class/devfreq/CIXH4000:00")
NPU_MHZ = (400, 600, 800, 1200)
COLS = ["trial", "job", "phase", "npu_mhz", "cpu_mhz", "gpu_mhz", "npu_ts", "npu_ms_per_tok", "npu_iters", "npu_est_GBps", "npu_busy",
        "cpu_ts", "cpu_ms_per_tok", "cpu_iters", "gpu_ts", "gpu_ms_per_tok", "gpu_proc_s", "work_iters", "work_tokens", "work_s",
        "heat_area_Cs", "heat_per_tok", "tail_s", "tail_complete", "temp_peak_c", "freq_ok", "temp_c", "temp_c_post"]
UTIL_COLS = ["npu_util", "npu_host_ms_per_tok", "npu_ttft_ms", "gpu_busy", "cpu_util_a720", "cpu_util_npu_host", "cpu_util_gpu_host",
             "cpu_util_all", "disk_util_nvme", "disk_util_sda", "disk_rd_MBps", "disk_wr_MBps", "read_GBps_a720", "bus_GBps_a720", "bus_GBps_a520"]
COLS[COLS.index("npu_busy") + 1:COLS.index("npu_busy") + 1] = UTIL_COLS
HEAT_JOBS = ("npu_heat",)
NPU_JOBS = ("npu_solo", "npu_cpu_solo", "npu_gpu_solo", "corun_npu_cpu", "corun_npu_gpu", "corun_npu_cpu_gpu", "npu_heat")
CPU_JOBS = ("cpu_solo", "corun_npu_cpu", "corun_npu_cpu_gpu")
GPU_JOBS = ("gpu_solo", "corun_npu_gpu", "corun_npu_cpu_gpu")
DEFAULT_GOV = 0            # n == 0 in a job tuple means "NPU on its default governor"


class Npu:
    """userspace governor + set_freq on the NPU devfreq node; everything is put back on restore()."""
    def __init__(self):
        self.gov = S.rd(NPU_DEV / "governor")
        try:
            self.set0 = S.rd(NPU_DEV / "userspace/set_freq")
        except Exception:      # noqa: BLE001
            self.set0 = ""
        self.cur0 = S.rd(NPU_DEV / "cur_freq")

    def pin(self, mhz):
        if S.rd(NPU_DEV / "governor") != "userspace":
            S.wr(NPU_DEV / "governor", "userspace")
        S.wr(NPU_DEV / "userspace/set_freq", mhz * 1000000)

    def cur_mhz(self):
        return int(S.rd(NPU_DEV / "cur_freq")) // 1000000

    def busy_ms(self):
        return int(S.rd(NPU_DEV / "device/power/runtime_active_time"))

    def restore(self):
        try:
            S.wr(NPU_DEV / "governor", self.gov)
            v = self.set0 if self.set0.isdigit() else self.cur0
            if (NPU_DEV / "userspace/set_freq").exists():
                S.wr(NPU_DEV / "userspace/set_freq", v)
        except Exception as e:     # noqa: BLE001
            print("npu restore failed:", e, file=sys.stderr)


class TempLog:
    """1 Hz trace of the hottest thermal zone."""
    def __init__(self):
        self.rows, self.stop = [], threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            self.rows.append((time.time(), S.temp_c()))
            self.stop.wait(1.0)

    def start(self):
        self.th.start()
        return self

    def end(self):
        self.stop.set()
        self.th.join(timeout=5)


def util_log(a):
    return U.UtilLog({"a720": U.cpulist(a.cpus), "npu_host": [a.npu_host_cpu], "gpu_host": [a.host_cpu], "all": None},
                     {"nvme": "nvme", "sda": "sda"}, perf_bin=a.perf, mode=getattr(a, "perf_mode", None), bus_bytes=a.bus_bytes,
                     tmp_dir=getattr(a, "out_dir_path", None))


def npu_extras(its, phase):
    """npu_util, npu_host_ms_per_tok, npu_ttft_ms from the NPU_RESULT extras of the counted iterations (s, e, tps, tokens, extras)."""
    ex = [x for x in its if len(x) > 4 and x[4].get("npu_ms") is not None]
    if not ex:
        return {}
    wall = sum((x[3] / x[2]) * 1000.0 for x in ex)                 # unit wall ms = tokens / tps
    npu_ms = sum(x[4]["npu_ms"] for x in ex)
    tok = sum(x[3] for x in ex)
    res = {"npu_util": npu_ms / wall if wall else None, "npu_host_ms_per_tok": (wall - npu_ms) / tok if tok else None}
    if phase == "prefill":
        res["npu_ttft_ms"] = med([(x[3] / x[2]) * 1000.0 for x in ex])
    return res


def heat_area(rows, t0_c, lo, hi):
    """Trapezoid integral of (T - t0_c) dt over [lo, hi] from the 1 Hz samples; also the peak in that window."""
    pts = [(t, T) for t, T in rows if lo <= t <= hi]
    if len(pts) < 2:
        return None, None
    area = sum((pts[i + 1][0] - pts[i][0]) * ((pts[i][1] + pts[i + 1][1]) / 2 - t0_c) for i in range(len(pts) - 1))
    return area, max(T for _, T in pts)


def host_cmd(a, phase, exec_=False):
    pin = "" if a.no_pin_npu_host else f"taskset -c {a.npu_host_cpu} "
    return ("exec " if exec_ else "") + pin + a.npu_cmd.replace("{phase}", phase)


def parse_result(line):
    """(tokens/s, tokens, extras); extras keeps npu_ms/host_ms when the command prints them (npu_llm_run.py does)."""
    d = json.loads(line.split(" ", 1)[1])
    return float(d["tps"]), int(d.get("tokens", 0)), {k: d[k] for k in ("npu_ms", "host_ms", "npu_mhz") if k in d}


def npu_once(a, phase):
    r = subprocess.run(host_cmd(a, phase), shell=True, capture_output=True, text=True, timeout=a.npu_timeout)
    for line in r.stdout.splitlines()[::-1]:
        if line.startswith("NPU_RESULT"):
            return parse_result(line)
    raise RuntimeError(f"no NPU_RESULT line (exit {r.returncode}): {(r.stderr or r.stdout)[-300:]}")


def npu_loop(a, phase, stop, out):
    """One process per iteration. out gets (start, end, tokens/s, tokens) or ("error", message)."""
    while not stop.is_set():
        try:
            s = time.time()
            ts, n, ex = npu_once(a, phase)
            out.append((s, time.time(), ts, n, ex))
        except Exception as e:     # noqa: BLE001
            out.append(("error", str(e)))
            return


def npu_loop_persistent(a, phase, stop, out):
    """One process for the whole job; every NPU_RESULT line it prints is one iteration. The first iteration starts at process start,
    so it carries the engine load (callers drop it). The process group is killed when `stop` is set."""
    errf = open(a.npu_err, "a") if getattr(a, "npu_err", None) else subprocess.DEVNULL
    p = subprocess.Popen(host_cmd(a, phase, exec_=True), shell=True, stdout=subprocess.PIPE, stderr=errf, text=True, start_new_session=True)

    def killer():
        stop.wait()
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    threading.Thread(target=killer, daemon=True).start()
    last = time.time()
    got = 0
    for line in p.stdout:
        if line.startswith("NPU_RESULT"):
            now = time.time()
            try:
                ts, n, ex = parse_result(line)
            except Exception as e:     # noqa: BLE001
                out.append(("error", f"bad NPU_RESULT line {line!r}: {e}"))
                break
            out.append((last, now, ts, n, ex))
            last = now
            got += 1
    p.wait()
    if not stop.is_set():
        out.append(("error", f"NPU process ended by itself after {got} results (exit {p.returncode})"))


def med(v):
    return statistics.median(v) if v else None


def loop_fn(a):
    return npu_loop_persistent if a.npu_persistent else npu_loop


def run_heat(a, npu, phase, t0):
    """npu_heat: --heat-iters inferences of fixed work, then the cool-down tail on the default governor. Returns (fields, detail)."""
    out, stop = [], threading.Event()
    tl = TempLog().start()
    ul = util_log(a).start()
    th = threading.Thread(target=loop_fn(a), args=(a, phase, stop, out), daemon=True)
    th.start()
    t_dead = time.time() + a.heat_max_s
    while len(out) < a.heat_iters + 1 and time.time() < t_dead and not (out and out[-1][0] == "error"):
        time.sleep(0.5)
    stop.set()
    th.join(timeout=a.npu_timeout + 30)
    t_work_end = time.time()
    ul.end()
    errs = [x for x in out if x and x[0] == "error"]
    if errs:
        tl.end()
        raise RuntimeError(f"npu loop: {errs[0][1]}")
    its = out[1:1 + a.heat_iters] if len(out) > 1 else []     # drop the first iteration (engine load, warm-up)
    if not its:
        tl.end()
        raise RuntimeError("npu_heat: no completed iteration")
    util = {**ul.window(its[0][0], its[-1][1]), **npu_extras(its, phase)}      # over the work only, not the tail
    t_start = its[0][0]
    work_end = its[-1][1]
    tokens = sum(x[3] for x in its)
    npu.restore()                                            # the tail always runs on the default governor
    tail0 = time.time()
    while S.temp_c() > t0 + a.tail_calm_c and time.time() - tail0 < a.tail_max:
        time.sleep(2)
    tail_ok = S.temp_c() <= t0 + a.tail_calm_c
    t_end = time.time()
    tl.end()
    area, peak = heat_area(tl.rows, t0, t_start, t_end)
    res = dict(work_iters=len(its), work_tokens=tokens, work_s=work_end - t_start, heat_area_Cs=area,
               heat_per_tok=(area / tokens) if (area is not None and tokens) else None, tail_s=t_end - work_end, tail_complete=int(tail_ok),
               temp_peak_c=peak, npu_ts=med([x[2] for x in its]), npu_iters=len(its), **util)
    if res["npu_ts"]:
        res["npu_ms_per_tok"] = 1000.0 / res["npu_ts"]
    if len(out) < a.heat_iters + 1:
        print(f"    WARNING: only {len(out) - 1} of {a.heat_iters} iterations finished within --heat-max-s", flush=True)
    return res, {"npu_iterations": [list(x) for x in out], "window": [t_start, t_end]}


def run_job(job, a, npu, phase, t0):
    """Returns (fields, raw detail). Background = NPU loop and/or CPU loop; foreground = GPU job or a timed window."""
    if job in HEAT_JOBS:
        return run_heat(a, npu, phase, t0)
    res = {}
    nout, cout, stop, stop_c = [], [], threading.Event(), threading.Event()
    nth = threading.Thread(target=loop_fn(a), args=(a, phase, stop, nout), daemon=True)
    cth = threading.Thread(target=C.cpu_loop, args=(a, stop_c, cout), daemon=True)
    with_npu, with_cpu, with_gpu = job in NPU_JOBS, job in CPU_JOBS, job in GPU_JOBS
    tl = TempLog().start()
    ul = util_log(a).start()
    if with_npu:
        nth.start()
        time.sleep(a.lead)
    if with_cpu:
        cth.start()
    b0 = (npu.busy_ms(), time.time()) if with_npu else None
    gs = ge = None
    if with_gpu:
        gs, ge, r = C.gpu_job(a)
        res.update(gpu_ts=r["avg_ts"], gpu_ms_per_tok=1000.0 / r["avg_ts"], gpu_proc_s=ge - gs)
    else:
        w_s = time.time()
        time.sleep(a.window)
        w_e = time.time()
    if b0 is not None:
        res["npu_busy"] = (npu.busy_ms() - b0[0]) / 1000.0 / (time.time() - b0[1])
    if with_npu:
        stop.set(); nth.join(timeout=a.npu_timeout + 30)
    if with_cpu:
        stop_c.set(); cth.join(timeout=120)
    tl.end()
    ul.end()
    lo, hi = (gs + a.guard, ge) if with_gpu else (w_s + a.guard, w_e)
    area, peak = heat_area(tl.rows, t0, lo, hi)
    res.update(heat_area_Cs=area, temp_peak_c=peak, **ul.window(lo, hi))
    for name, out in (("npu", nout), ("cpu", cout)):
        errs = [x for x in out if x and x[0] == "error"]
        if errs:
            raise RuntimeError(f"{name} loop: {errs[0][1]}")
    ni, ci = list(nout), list(cout)
    detail = {"npu_iterations": [list(x) for x in ni], "cpu_iterations": [list(x) for x in ci], "window": [lo, hi]}
    if with_npu:
        if job.endswith("_solo"):
            use = ni[1:] if len(ni) > 1 else ni              # first iteration carries engine load and warm-up
        else:
            use = [x for x in ni if x[0] >= lo and x[1] <= hi]
        if use:
            ts = med([x[2] for x in use])
            res.update(npu_ts=ts, npu_ms_per_tok=1000.0 / ts, npu_iters=len(use),
                       npu_est_GBps=(a.npu_model_bytes * ts / 1e9 if phase == "decode" and a.npu_model_bytes else None),
                       **npu_extras(use, phase))
    if with_cpu:
        use = ci[1:] if job == "cpu_solo" and len(ci) > 1 else [x for x in ci if x[0] >= lo and x[1] <= hi]
        if use:
            ts = med([x[2] for x in use])
            res.update(cpu_ts=ts, cpu_ms_per_tok=1000.0 / ts, cpu_iters=len(use))
    return res, detail


# ------------------------------------------------------------------ summary and slowdowns (no board needed)
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def nz(x):
    """normalise a frequency cell to a string key: '400', '2200', '' for none."""
    v = num(x)
    return "" if v is None else str(int(v))


def analyze(csv_path, out_path):
    """summary.csv from a sweep.csv: mean over trials per (job, phase, npu, cpu, gpu), then slowdowns against the matching solo rows."""
    allrows = list(csv.DictReader(open(csv_path, newline="")))
    rows = [r for r in allrows if r.get("freq_ok") == "1"]
    dropped = len(allrows) - len(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["job"], r["phase"], nz(r["npu_mhz"]), nz(r["cpu_mhz"]), nz(r["gpu_mhz"]))].append(r)
    means = ("npu_ts", "cpu_ts", "gpu_ts", "npu_est_GBps", "npu_busy", *UTIL_COLS, "heat_area_Cs", "heat_per_tok", "work_s", "tail_s",
             "tail_complete", "temp_peak_c")
    agg = {}
    for k, rs in groups.items():
        d = {"n": len(rs)}
        for c in means:
            v = [num(r.get(c)) for r in rs]
            v = [x for x in v if x is not None]
            d[c] = statistics.mean(v) if v else None
        agg[k] = d

    def find(job, phase, **sel):
        for (j, ph, n, c, g), d in agg.items():
            if j == job and ph == phase and all({"npu": n, "cpu": c, "gpu": g}[k] == v for k, v in sel.items()):
                return d
        return None

    out = []
    for (job, phase, n, c, g), d in sorted(agg.items()):
        row = {"job": job, "phase": phase, "npu_mhz": n, "cpu_mhz": c, "gpu_mhz": g, **d}
        if job.startswith("corun"):
            refs = {"npu": ("npu_ts", find("npu_solo", phase, npu=n) if n else None),
                    "cpu": ("cpu_ts", find("cpu_solo", phase, cpu=c) if c else None),
                    "gpu": ("gpu_ts", find("gpu_solo", phase, gpu=g) if g else None)}
            for x, (col, ref) in refs.items():
                ts, rts = d.get(col), (ref or {}).get(col)
                if ts and rts:
                    row[f"{x}_slowdown"] = rts / ts - 1
                    row[f"{x}_speed_loss"] = 1 - ts / rts
            if job == "corun_npu_cpu_gpu":
                for x, col, pair in (("npu", "npu_ts", ("corun_npu_cpu", {"npu": n, "cpu": c})), ("npu", "npu_ts", ("corun_npu_gpu", {"npu": n, "gpu": g})),
                                     ("cpu", "cpu_ts", ("corun_npu_cpu", {"npu": n, "cpu": c})), ("gpu", "gpu_ts", ("corun_npu_gpu", {"npu": n, "gpu": g}))):
                    p = find(pair[0], phase, **pair[1])
                    if p and p.get(col) and d.get(col):
                        row[f"{x}_extra_vs_{pair[0].split('_')[-1]}_pair"] = 1 - d[col] / p[col]
        out.append(row)
    cols = ["job", "phase", "npu_mhz", "cpu_mhz", "gpu_mhz", "n"] + list(means) + [f"{x}_{k}" for x in ("npu", "cpu", "gpu") for k in ("slowdown", "speed_loss")] \
        + ["npu_extra_vs_cpu_pair", "npu_extra_vs_gpu_pair", "cpu_extra_vs_cpu_pair", "gpu_extra_vs_gpu_pair"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow({k: S.fmt(r.get(k)) for k in cols})
    print(f"summary: {out_path} ({len(out)} groups from {len(rows)} rows; {dropped} rows dropped for freq_ok=0)")
    for r in out:
        if r["job"].startswith("corun"):
            bits = [f"{x}: {-r[f'{x}_speed_loss'] * 100:+.1f}% speed" for x in ("npu", "cpu", "gpu") if r.get(f"{x}_speed_loss") is not None]
            print(f"  {r['job']:18s} npu={r['npu_mhz'] or '-':>4s} cpu={r['cpu_mhz'] or '-':>4s} gpu={r['gpu_mhz'] or '-':>4s}  " + "  ".join(bits))
    return out


def main():
    ap = S.build_parser(__doc__, freqs="1200,1500,1800,2200", freqs_help="A720 MHz list (real steps; 2200 = common ceiling)")
    ap.add_argument("--npu-cmd", help="command that runs the NPU LLM and prints NPU_RESULT {...}; {phase} = prefill|decode (required unless --analyze)")
    ap.add_argument("--npu-persistent", action="store_true", help="start --npu-cmd once per job; it prints one NPU_RESULT line per inference until killed")
    ap.add_argument("--analyze", metavar="SWEEP_CSV", help="only write summary.csv (next to the CSV) from an existing sweep.csv, then exit")
    ap.add_argument("--phase", default="decode", choices=["prefill", "decode"], help="phase of the NPU job (CPU/GPU sides always decode)")
    ap.add_argument("--npu-freqs", default="400,600,800,1200")
    ap.add_argument("--npu-coru-freqs", default="400,800,1200", help="NPU MHz for the co-run grids")
    ap.add_argument("--gpu-freqs", default="350,600,800,1000")
    ap.add_argument("--gpu-coru-freqs", default="350,600,1000")
    ap.add_argument("--cpu-coru-freqs", default="1200,2200")
    ap.add_argument("--no-tri", action="store_true", help="skip the three-way co-run job")
    ap.add_argument("--jobs", default=None, help="comma list of job types to keep (e.g. npu_solo,cpu_solo,gpu_solo,corun_npu_cpu,corun_npu_gpu)")
    ap.add_argument("--tri-npu-freqs", default="400,1200", help="three-way grid; every value must also be in the pair grids (for *_extra_vs_pair)")
    ap.add_argument("--tri-cpu-freqs", default="1200,2200")
    ap.add_argument("--tri-gpu-freqs", default="350,1000")
    ap.add_argument("--with-heat", action="store_true", help="add the fixed-work npu_heat jobs (NPU thermal A/B); long: work + cool-down tail each")
    ap.add_argument("--heat-iters", type=int, default=100, help="inferences of fixed work per npu_heat job (choose ~5 min at the top frequency)")
    ap.add_argument("--heat-max-s", type=float, default=1800.0, help="give up the work after this many seconds")
    ap.add_argument("--tail-max", type=float, default=600.0, help="longest cool-down tail of an npu_heat job")
    ap.add_argument("--tail-calm-c", type=float, default=1.0, help="the tail ends when T <= T_start + this")
    ap.add_argument("--cpu-solo-mhz", type=int, default=2200, help="A720 MHz while the NPU runs alone / the GPU job runs")
    ap.add_argument("--npu-host-cpu", type=int, default=3, help="core of the NPU host process (an A520, not cpu2 = the GPU host)")
    ap.add_argument("--no-pin-npu-host", action="store_true", help="do not taskset the NPU command (tests off the board)")
    ap.add_argument("--npu-model-bytes", type=float, default=0, help="weight bytes read per decoded token, for the traffic estimate (0 = skip)")
    ap.add_argument("--npu-timeout", type=float, default=300.0)
    ap.add_argument("--gpu-tg", type=int, default=64)
    ap.add_argument("--gpu-reps", type=int, default=8)
    ap.add_argument("--cpu-iter-reps", type=int, default=3)
    ap.add_argument("--host-cpu", type=int, default=2, help="core of the GPU job's host thread (an A520)")
    ap.add_argument("--host-mhz", type=int, default=1800)
    ap.add_argument("--lead", type=float, default=5.0, help="seconds the NPU loop runs before the other job starts")
    ap.add_argument("--guard", type=float, default=6.0)
    ap.add_argument("--window", type=float, default=40.0, help="seconds of a timed (non-GPU) job")
    ap.set_defaults(trials=3)
    a = ap.parse_args()
    if a.analyze:
        p = Path(a.analyze)
        analyze(p, p.with_name("summary.csv"))
        return
    if not a.npu_cmd:
        ap.error("--npu-cmd is required (unless --analyze)")
    ints = lambda s: [int(x) for x in s.split(",")]      # noqa: E731
    nf, ncf, gf, gcf, cf, ccf = (ints(a.npu_freqs), ints(a.npu_coru_freqs), ints(a.gpu_freqs), ints(a.gpu_coru_freqs),
                                 ints(a.freqs), ints(a.cpu_coru_freqs))
    tn, tc, tg = ints(a.tri_npu_freqs), ints(a.tri_cpu_freqs), ints(a.tri_gpu_freqs)
    top = max(nf)
    jobs = ([("npu_solo", n, a.cpu_solo_mhz, None) for n in nf]
            + [("npu_cpu_solo", top, c, None) for c in cf] + [("npu_gpu_solo", top, a.cpu_solo_mhz, g) for g in gf]
            + [("cpu_solo", None, c, None) for c in ccf] + [("gpu_solo", None, a.cpu_solo_mhz, g) for g in gcf]
            + [("corun_npu_cpu", n, c, None) for n in ncf for c in ccf] + [("corun_npu_gpu", n, a.cpu_solo_mhz, g) for n in ncf for g in gcf])
    if not a.no_tri:
        jobs += [("corun_npu_cpu_gpu", n, c, g) for n in tn for c in tc for g in tg]
    if a.with_heat:
        jobs += [("npu_heat", n, a.cpu_solo_mhz, None) for n in nf + [DEFAULT_GOV]]
    if a.jobs:
        keep = set(a.jobs.split(","))
        bad = keep - {j[0] for j in jobs} - set(HEAT_JOBS) - {"corun_npu_cpu_gpu"}
        if bad:
            ap.error(f"--jobs: unknown job type(s) {sorted(bad)}")
        jobs = [j for j in jobs if j[0] in keep]
    print(f"plan: {len(jobs)} jobs per trial x {a.trials} trials, NPU phase={a.phase}, "
          f"{'persistent' if a.npu_persistent else 'one process per iteration'} NPU command")
    for j in jobs:
        print("  ", j)
    if not a.npu_persistent:
        print("NOTE: without --npu-persistent every co-run iteration pays the engine load (the NPU idles), so the interference is underestimated.")
    if a.phase == "decode" and not a.npu_model_bytes:
        print("NOTE: --npu-model-bytes not given: npu_est_GBps (the NPU traffic estimate) will be empty.")
    if a.dry_run:
        return
    if platform.machine() != "aarch64":
        sys.exit("run this on the board")
    for p in (a.model, a.cpu_bench, a.gpu_bench):
        if not Path(p).exists():
            sys.exit(f"missing: {p}")
    if not os.environ.get("SUDO_PASS") and subprocess.run(["sudo", "-n", "true"]).returncode:
        sys.exit("passwordless sudo needed (frequency pinning)")
    out_dir = Path(a.out_dir) if a.out_dir else S.REPO / "runs" / time.strftime("npu_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    a.npu_err = str(out_dir / "npu_stderr.log")
    a.out_dir_path = str(out_dir)
    a.perf_mode = U.perf_mode(a.perf)
    print("bus_GBps (perf bus_access):", U.perf_note(a.perf_mode), flush=True)
    (out_dir / "meta.json").write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "kernel": platform.release(),
                                                   "temp_start_c": S.temp_c()}, indent=1))
    fq, host, npu = S.Freq(), C.HostPin(a.host_cpu), Npu()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    raw = open(out_dir / "raw.jsonl", "w")
    fh = open(out_dir / "sweep.csv", "w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    cpu_list = [int(c) for c in a.cpus.split(",")]
    try:
        host.pin(a.host_mhz * 1000)
        for trial in range(1, a.trials + 1):
            order = jobs[:]
            random.Random(trial).shuffle(order)
            for job, n, c, g in order:
                fq.pin_cpu(c * 1000)
                fq.pin_gpu(g * 1000000) if g else fq.unpin_gpu()
                if n == DEFAULT_GOV:
                    npu.restore()                       # NPU on its default governor (npu_heat baseline)
                else:
                    npu.pin(n if n else top)
                time.sleep(1.0)
                ok, _ = S.verify_cpu(c * 1000, cpu_list)
                ok = ok and (n in (None, DEFAULT_GOV) or npu.cur_mhz() == n) and (g is None or int(S.rd(S.GPU_DEV / "cur_freq")) // 1000000 == g)
                S.cooldown(a.cool, a.cool_max)
                t0 = S.temp_c()
                print(f"[trial {trial}] {job:18s} npu={n} cpu={c} gpu={g} freq_ok={ok} temp={t0:.0f}C", flush=True)
                res, detail = run_job(job, a, npu, a.phase, t0)
                raw.write(json.dumps({"trial": trial, "job": job, "npu_mhz": n, "cpu_mhz": c, "gpu_mhz": g, **detail}) + "\n"); raw.flush()
                row = {"trial": trial, "job": job, "phase": a.phase, "npu_mhz": n, "cpu_mhz": c, "gpu_mhz": g, **res, "freq_ok": int(ok),
                       "temp_c": round(t0, 1), "temp_c_post": round(S.temp_c(), 1)}
                w.writerow({k: S.fmt(row.get(k)) for k in COLS}); fh.flush()
                print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    except KeyboardInterrupt:
        print("interrupted; restoring. Partial results are in", out_dir)
    finally:
        fq.restore(); host.restore(); npu.restore()
        fh.close(); raw.close()
    try:
        analyze(out_dir / "sweep.csv", out_dir / "summary.csv")
    except Exception as e:     # noqa: BLE001
        print("summary failed (rebuild with --analyze):", e, file=sys.stderr)


if __name__ == "__main__":
    main()
