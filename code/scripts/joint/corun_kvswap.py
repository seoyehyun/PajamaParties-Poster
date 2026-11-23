#!/usr/bin/env python3
"""corun_kvswap.py -- with a GPU decode running beside it, does a KV-swap CPU decode interfere less than a RAM-KV CPU decode?

Plan and reasoning: docs/joint/corun-kvswap-plan.md. Earlier co-run results: a CPU decode beside a GPU decode slowed the GPU by up to
+75% and the A720 frequency set how much (docs/sweep/results/cpu-gpu-memory-sweeps-20261113.md, 3; docs/joint/results/
thermal-ab-co-20261118.md). Those CPU jobs kept the KV in RAM at short context. KV swap reads only the selected KV groups, so at long
context a CPU decode should pull less from DRAM. Does that shrink the GPU's slowdown?

One CPU PROCESS per (trial, KV mode, A720 MHz): scripts/kvswap/run_stage1.sh on a long prompt, A720 x8 threads, the A720 policies
pinned for the whole process (or left on the default governor with "def").
  ram    run_stage1.sh base    (KV in RAM, full attention)
  swap   run_stage1.sh select  (KV on NVMe, direct I/O, predictor adapter, reuse buffer)
After the prefill, while that process decodes, a fixed schedule runs (GPU frequency order shuffled per process):

    settle | solo | GPU job @g1 | solo | GPU job @g2 | solo

  solo      the CPU decode alone for --solo-s seconds (GPU idle)
  GPU job   llama-bench, Vulkan build, -ngl 99, ONE host thread on --host-cpu (an A520 pinned to --host-mhz), tg --gpu-tg x --gpu-reps,
            GPU devfreq pinned to g. Its CPU window is [job start + --guard, job end] (the guard skips the GPU job's model load).
  solo_gpu  the same GPU job with the CPU side idle: one unit per GPU frequency and trial (the GPU baseline)
The CPU process is stopped after the last solo window (--let-finish waits for it instead, which also prints llama's own timing).

CPU speed per window comes from the generated tokens: out.txt is made a FIFO that this script reads, and llama-completion flushes every
token, so each read is normally one token and gets a timestamp (tokens.csv). CHECK THIS ON THE BOARD FIRST (plan, step P1): with
--let-finish the number of reads should equal llama's generated-token count. CPU slowdown of a GPU window = its ms/token vs the mean of
the solo windows right before and after it (same process, so the slow drift of the decode as its context grows cancels). GPU slowdown
= its ms/token vs solo_gpu at the same GPU frequency in the same trial. Noise floor = spread of the solo windows inside one process.
With SUDO_PASS set, `perf stat -I 1000 -C <A720 cores>` logs l2d_cache_refill x 64 B per second: the A720 read-bandwidth proxy that
was calibrated in docs/sweep/results/cpu-gpu-memory-sweeps-20261113.md (4). Its timestamps can lag the window clock by up to ~1 s.

Needs: the board, ~/llama-base/build/bin/llama-completion (KV swap patches 01-05), ~/llama-base/build-vulkan/bin/llama-bench, the prompt
and adapter files, passwordless `sudo -n tee` (frequencies, drop_caches; same sudoers rules as thermal_ab.py). SUDO_PASS only for perf.
  python3 scripts/joint/corun_kvswap.py --dry-run                         # the plan and the time estimate (also works off the board)
  SUDO_PASS=... python3 scripts/joint/corun_kvswap.py                     # full run (with the A720 bandwidth log)
  python3 scripts/joint/corun_kvswap.py --trials 1 --modes swap --a720 1500 --gpu-freqs 1000 --let-finish    # pilot P3
  python3 scripts/joint/corun_kvswap.py --resume runs/corunkv_<ts> <same flags>   # after a reboot: finished units are skipped
  python3 scripts/joint/corun_kvswap.py --analyze runs/corunkv_<ts>               # recompute windows.csv and summary.csv
Frequencies are restored on exit / Ctrl-C / SIGTERM. Raw output: runs/corunkv_<timestamp>/ (git-ignored).
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import re
import select
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # scripts/joint/
sys.path.insert(0, str(HERE.parent / "sweep"))            # sweep_cpufreq and sweep_corun (helpers) live in scripts/sweep/
import sweep_cpufreq as S   # noqa: E402
import sweep_corun as C     # noqa: E402  (HostPin: pins the A520 policy of the GPU job's host thread)

REPO = S.REPO
GPU_SOLO_MS = {350: 99.7, 600: 59.8, 800: 44.4, 1000: 37.7}   # GPU decode ms/token alone (11/13 sweeps), for the time estimate only
KEY_ARGS = ("modes", "a720", "gpu_freqs", "trials", "prompt_chars", "ctx", "ntok", "cpus", "threads", "m", "g", "w", "sink", "reuse",
            "gpu_tg", "gpu_reps", "host_cpu", "host_mhz", "solo_s", "guard", "settle")
WIN_COLS = ["trial", "mode", "a720", "gpu_mhz", "gpu_pos", "cpu_solo_ms", "cpu_corun_ms", "cpu_slowdown", "cpu_tokens", "gpu_solo_ms",
            "gpu_corun_ms", "gpu_slowdown", "a720_GBps_solo", "a720_GBps_corun", "gpu_est_GBps", "temp_max_c", "a720_ok", "gpu_ok", "note"]
SUM_COLS = ["mode", "a720", "gpu_mhz", "n", "cpu_solo_ms", "cpu_corun_ms", "cpu_slowdown", "gpu_solo_ms", "gpu_corun_ms", "gpu_slowdown",
            "a720_GBps_solo", "a720_GBps_corun", "gpu_est_GBps", "temp_max_c"]


# ------------------------------------------------------------------ small helpers
def cpu_ids(spec):
    """'0-1,6-11' -> [0, 1, 6, 7, 8, 9, 10, 11]"""
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def mlabel(mhz):
    return "def" if mhz is None else str(mhz)


def parse_a720(spec):
    return [None if x == "def" else int(x) for x in spec.split(",")]


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def est_gpu_job_s(a, g):
    return 8.0 + a.gpu_tg * a.gpu_reps * GPU_SOLO_MS.get(g, 37700.0 / g) * 1.5 / 1000      # model load + decode, co-run up to ~1.5x slower


def est_schedule_s(a, gfs):
    return a.settle + (len(gfs) + 1) * a.solo_s + sum(est_gpu_job_s(a, g) for g in gfs)


def est_prefill_s(a, mhz):
    # 7,013 prompt tokens took 333 s on the default governor and 501 s at 1500 MHz (thermal-ab-20261113): about 21 tok/s x f/2200
    return a.prompt_chars / a.chars_per_token / (21.0 * min(mhz or 2200, 2200) / 2200)


def build_parser():
    h = Path.home()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="ram,swap", help="CPU job KV mode(s): ram (run_stage1.sh base) and/or swap (select + predictor + reuse)")
    ap.add_argument("--a720", default="1500,def", help="A720 MHz of the CPU process; def = default governor. 800 and 1500 are the points where "
                                                       "request = real frequency on all four policies (board-facts-20261113)")
    ap.add_argument("--gpu-freqs", default="600,1000", help="GPU MHz list (available: 72 216 350 600 800 1000)")
    ap.add_argument("--trials", type=int, default=2, help="the unit order is shuffled per trial")
    ap.add_argument("--prompt-src", default="/mnt/ssd/prompts/haystack_c4.txt")
    ap.add_argument("--prompt-chars", type=int, default=32000, help="first N characters (32000 = 7,013 tokens, as in thermal-ab-20261113)")
    ap.add_argument("--chars-per-token", type=float, default=4.56, help="for the estimates and the context check only")
    ap.add_argument("--ctx", type=int, default=12288)
    ap.add_argument("--ntok", type=int, default=0, help="tokens the CPU process may generate; 0 = enough for the schedule at --min-cpu-ms")
    ap.add_argument("--min-cpu-ms", type=float, default=55.0, help="fastest CPU decode expected (ms/token); sizes --ntok when it is 0")
    ap.add_argument("--model", default=str(h / "PajamaParties/models/Qwen3-1.7B-Q4_K_M.gguf"))
    ap.add_argument("--llama-bin", default=str(h / "llama-base/build/bin/llama-completion"))
    ap.add_argument("--run-stage1", default=str(HERE.parent / "kvswap" / "run_stage1.sh"))
    ap.add_argument("--gpu-bench", default=str(h / "llama-base/build-vulkan/bin/llama-bench"))
    ap.add_argument("--cpus", default="0-1,6-11", help="A720 cores of the CPU process (taskset list)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--kv-file", default="/mnt/ssd/kv.bin")
    ap.add_argument("--dev", default="nvme0n1")
    ap.add_argument("--adapter", default="/mnt/ssd/adapters/kvswap-adapter-r128.gguf", help="predictor adapter; '' = uniform selection")
    ap.add_argument("--m", type=int, default=50, help="KV groups selected")
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--reuse", type=int, default=200, help="LLAMA_KV_REUSE groups per layer (0 = off)")
    ap.add_argument("--io-threads", type=int, default=4)
    ap.add_argument("--gpu-tg", type=int, default=64, help="decode tokens per rep of the GPU job")
    ap.add_argument("--gpu-reps", type=int, default=4, help="reps of the GPU job (one llama-bench process)")
    ap.add_argument("--host-cpu", type=int, default=2, help="core of the GPU job's host thread (an A520)")
    ap.add_argument("--host-mhz", type=int, default=1800, help="MHz that core's policy is pinned to")
    ap.add_argument("--guard", type=float, default=6.0, help="seconds at the start of a GPU job left out of its CPU window (model load)")
    ap.add_argument("--settle", type=float, default=10.0, help="seconds of decode before the first window")
    ap.add_argument("--solo-s", type=float, default=20.0, help="length of each solo window")
    ap.add_argument("--prefill-timeout", type=float, default=2400.0, help="give up on a CPU process whose first token takes longer")
    ap.add_argument("--let-finish", action="store_true", help="let the CPU process generate all --ntok tokens instead of stopping it")
    ap.add_argument("--perf", default="/usr/lib/linux-tools/6.8.0-147-generic/perf")
    ap.add_argument("--no-pmu", action="store_true", help="no A720 bandwidth log (no SUDO_PASS needed)")
    ap.add_argument("--cool", type=float, default=50.0, help="wait until temp_max <= this (C) before a unit")
    ap.add_argument("--cool-max", type=float, default=180.0, help="but at most this many seconds")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", default=None, help="continue an interrupted run directory (same flags as before)")
    ap.add_argument("--analyze", default=None, help="recompute windows.csv and summary.csv of an existing run directory")
    ap.add_argument("--dry-run", action="store_true")
    return ap


# ------------------------------------------------------------------ live logging
class TokenTap:
    """out.txt as a FIFO: run_stage1.sh redirects llama-completion's stdout into it, a thread here timestamps every read.
    The script keeps its own idle write end open, so the reader sees no EOF before the real writer has come and gone."""
    def __init__(self, fifo, copy_path):
        os.mkfifo(fifo)
        self.rfd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        self.wfd = os.open(fifo, os.O_WRONLY)
        self.rows = []
        self.copy = open(copy_path, "wb")
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while True:
            r, _, _ = select.select([self.rfd], [], [], 0.5)
            if not r:
                continue
            try:
                d = os.read(self.rfd, 65536)
            except BlockingIOError:
                continue
            if not d:                                  # every writer is gone (ours is closed in close())
                break
            self.rows.append((time.time(), len(d)))
            self.copy.write(d)
        self.copy.close()

    def first(self):
        return self.rows[0][0] if self.rows else None

    def close(self):
        if self.wfd is not None:
            os.close(self.wfd)
            self.wfd = None
        self.th.join(timeout=15)
        os.close(self.rfd)


class TempLog:
    def __init__(self):
        self.rows, self.stop = [], threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            self.rows.append((time.time(), S.temp_c()))
            self.stop.wait(1.0)

    def start(self):
        self.th.start()

    def end(self):
        self.stop.set()
        self.th.join(timeout=5)


class PmuLog:
    """sudo perf stat -I 1000 on the A720 cores, l2d_cache_refill only; stderr (perf's output) goes to a file of ours.
    perf runs under `sleep <cap>`, so it ends by itself even if the kill below fails."""
    def __init__(self, a, path, pidfile, cap_s):
        self.p, self.t0, self.err, self.pidfile = None, None, None, pidfile
        if a.no_pmu:
            return
        pw = os.environ.get("SUDO_PASS")
        self.direct = S.perf_direct()               # perf_event_paranoid <= 0: no sudo needed
        if not pw and not self.direct:
            print("   note: neither perf_event_paranoid <= 0 nor SUDO_PASS: no A720 bandwidth log for this unit", flush=True)
            return
        inner = (f"echo $$ > {pidfile}; exec {a.perf} stat -I 1000 -x, -C {a.cpus} -e {S.PMU}/l2d_cache_refill/ -- sleep {int(cap_s)}")
        self.err = open(path, "w")
        if self.direct:
            self.p = subprocess.Popen(["sh", "-c", inner], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self.err)
        else:
            self.p = subprocess.Popen(["sudo", "-S", "-p", "", "sh", "-c", inner], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      stderr=self.err, text=True)
            self.p.stdin.write(pw + "\n")
            self.p.stdin.close()
        self.t0 = time.time()

    def stop(self):
        if not self.p:
            return
        if self.p.poll() is None:
            try:
                pid = Path(self.pidfile).read_text().strip()
                if self.direct:
                    os.kill(int(pid), signal.SIGINT)
                else:
                    S.sudo_run(["kill", "-INT", pid])
            except (OSError, ValueError):
                pass
            try:
                self.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print("   warning: perf did not stop; it ends by itself after its cap", flush=True)
        self.err.close()


# ------------------------------------------------------------------ units
def gpu_job(a):
    ga = argparse.Namespace(cpus=str(a.host_cpu), model=a.model, threads=1, pp=0, tg=a.gpu_tg, no_pmu=True)
    t0 = time.time()
    r = S.run_llama(a.gpu_bench, 99, "tg", a.gpu_reps, ga, perf=False)
    return {"t0": t0, "t1": time.time(), "avg_ts": r["avg_ts"], "sd_ts": r["sd_ts"], "model_size": r["model_size"]}


def pin_gpu(fq, g):
    fq.pin_gpu(g * 1000000)
    time.sleep(0.5)
    return int(S.rd(S.GPU_DEV / "cur_freq")) // 1000000 == g


def set_a720(fq, mhz, a):
    """None = default governor (not verified). NOTE: verify_cpu reads cpuinfo_cur_freq, which only echoes the request on this board."""
    fq.restore()
    if mhz is None:
        return None
    fq.pin_cpu(mhz * 1000)
    time.sleep(1.0)
    return S.verify_cpu(mhz * 1000, a.cpu_ids)[0]


def run_gpu_unit(trial, g, a, fq):
    S.cooldown(a.cool, a.cool_max)
    fq.restore()
    u = {"unit": "solo_gpu", "trial": trial, "gpu_mhz": g, "temp_c": S.temp_c()}
    try:
        u["gpu_ok"] = pin_gpu(fq, g)
        u["gpu"] = gpu_job(a)
        print(f"[trial {trial}] solo_gpu {g} MHz: {1000 / u['gpu']['avg_ts']:.1f} ms/tok", flush=True)
    except RuntimeError as e:
        u["error"] = str(e)
        print(f"[trial {trial}] solo_gpu {g} MHz FAILED: {e}", flush=True)
    finally:
        fq.restore()
    return u


def run_cpu_unit(trial, mode, mhz, gorder, a, fq, out_dir, prompt):
    tag = f"t{trial}_{mode}_c{mlabel(mhz)}"
    rdir = out_dir / tag
    shutil.rmtree(rdir, ignore_errors=True)            # a resumed run may find the partial files of an interrupted attempt
    rdir.mkdir(parents=True)
    u = {"unit": "cpu", "trial": trial, "mode": mode, "a720": mhz, "gpu_order": gorder, "dir": tag, "windows": []}
    S.cooldown(a.cool, a.cool_max)
    u["temp_c"] = S.temp_c()
    u["a720_ok"] = set_a720(fq, mhz, a)
    env = {**os.environ, "LLAMA_BIN": a.llama_bin, "MODEL": a.model, "PROMPT": str(prompt), "KV_FILE": a.kv_file, "CTX": str(a.ctx),
           "NTOK": str(a.ntok), "CPUS": a.cpus, "THREADS": str(a.threads), "DEV": a.dev, "OUTDIR": str(rdir)}
    if mode == "swap":
        env.update(M=str(a.m), G=str(a.g), W=str(a.w), SINK=str(a.sink), PREFETCH="1", KV_IO="direct", IO_THREADS=str(a.io_threads),
                   LLAMA_KV_REUSE=str(a.reuse))
        if a.adapter:
            env["ADAPTER"] = a.adapter
    tap = TokenTap(rdir / "out.txt", rdir / "gen.txt")
    temps = TempLog()
    temps.start()
    pmu = None
    print(f"[trial {trial}] cpu {mode} A720 {mlabel(mhz)} (verified {u['a720_ok']}), GPU order {gorder}: prefill ...", flush=True)
    t_launch = time.time()
    logf = open(rdir / "run_stage1.log", "w")
    proc = subprocess.Popen(["bash", a.run_stage1, "base" if mode == "ram" else "select"], env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        while tap.first() is None:
            if proc.poll() is not None:
                raise RuntimeError(f"CPU process ended (rc {proc.returncode}) before its first token: see {rdir}/err.txt")
            if time.time() - t_launch > a.prefill_timeout:
                raise RuntimeError(f"no first token after {a.prefill_timeout:.0f} s")
            time.sleep(0.2)
        u["decode_start"] = tap.first()
        u["prefill_wall_s"] = tap.first() - t_launch
        print(f"   first token after {u['prefill_wall_s']:.0f} s; schedule starts", flush=True)
        pmu = PmuLog(a, rdir / "pmu.txt", rdir / "perf.pid", cap_s=2 * est_schedule_s(a, gorder) + 60)
        u["pmu_t0"] = pmu.t0
        time.sleep(a.settle)
        for g in gorder + [None]:
            if proc.poll() is not None:
                u["note"] = "CPU process ended before the schedule did: raise --ntok"
                break
            w0 = time.time()
            time.sleep(a.solo_s)
            u["windows"].append({"kind": "solo", "t0": w0, "t1": time.time(), "alive": proc.poll() is None})
            if g is None or proc.poll() is not None:
                continue
            ok = pin_gpu(fq, g)
            job = gpu_job(a)
            fq.unpin_gpu()
            u["windows"].append({"kind": "corun", "gpu_mhz": g, "gpu_ok": ok, "t0": job["t0"] + a.guard, "t1": job["t1"], "gpu": job,
                                 "alive": proc.poll() is None})
            print(f"   GPU {g} MHz: {1000 / job['avg_ts']:.1f} ms/tok ({job['t1'] - job['t0']:.0f} s)", flush=True)
    except RuntimeError as e:
        u["error"] = str(e)
        print(f"   FAILED: {e}", flush=True)
    finally:
        if pmu:
            pmu.stop()
        if a.let_finish and "error" not in u:
            proc.wait()
        elif proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)       # run_stage1.sh, llama-completion, iostat, mpstat share the session
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        logf.close()
        tap.close()
        temps.end()
        fq.restore()
    u["rc"] = proc.returncode
    u["n_reads"] = len(tap.rows)
    with open(rdir / "tokens.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "bytes"])
        w.writerows((f"{t:.4f}", n) for t, n in tap.rows)
    with open(rdir / "temps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "temp_c"])
        w.writerows((f"{t:.2f}", c) for t, c in temps.rows)
    err = (rdir / "err.txt").read_text(errors="replace") if (rdir / "err.txt").exists() else ""
    ev = re.search(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs", err)
    pe = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", err)
    if ev:                                             # only when the process finished on its own (--let-finish)
        u.update(llama_decode_runs=int(ev.group(2)), llama_tpot_ms=float(ev.group(1)) / int(ev.group(2)))
    if pe:
        u["prompt_tokens"] = int(pe.group(2))
    return u


# ------------------------------------------------------------------ analysis
def read_csv_pairs(path):
    if not path.exists():
        return []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        return [(float(x), float(y)) for x, y in r]


def read_pmu(path, t0):
    """perf -I -x, lines 'time,count,unit,event,...' -> [(wall time at the end of the interval, GB/s in it)]"""
    out, prev = [], 0.0
    if t0 is None or not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        p = line.split(",")
        if len(p) < 4 or "l2d_cache_refill" not in p[3]:
            continue
        try:
            t, c = float(p[0]), int(p[1])
        except ValueError:
            continue
        if t > prev:
            out.append((t0 + t, c * 64 / (t - prev) / 1e9))
        prev = t
    return out


def cpu_ms(tokens, t0, t1):
    ts = [t for t, _ in tokens if t0 <= t <= t1]
    return (1000.0 * (ts[-1] - ts[0]) / (len(ts) - 1), len(ts)) if len(ts) >= 3 else (None, len(ts))


def window_bw(pmu, t0, t1):
    return mean([bw for t, bw in pmu if t0 + 1.0 <= t <= t1])    # intervals that end inside the window, past its first second


def analyze(out_dir):
    units = [json.loads(line) for line in open(out_dir / "units.jsonl")]
    gsolo = {}
    for u in units:
        if u["unit"] == "solo_gpu" and "gpu" in u:
            gsolo.setdefault((u["trial"], u["gpu_mhz"]), []).append(1000.0 / u["gpu"]["avg_ts"])
    rows, noise, checks = [], [], []
    for u in units:
        if u["unit"] != "cpu" or not u.get("windows"):
            continue
        d = out_dir / u["dir"]
        tokens = read_csv_pairs(d / "tokens.csv")
        temps = read_csv_pairs(d / "temps.csv")
        pmu = read_pmu(d / "pmu.txt", u.get("pmu_t0"))
        if u.get("llama_decode_runs"):
            checks.append((u["dir"], u["n_reads"], u["llama_decode_runs"] + 1))
        ws = u["windows"]
        for w in ws:
            w["cpu_ms"], w["cpu_n"] = cpu_ms(tokens, w["t0"], w["t1"])
            w["bw"] = window_bw(pmu, w["t0"], w["t1"])
            w["tmax"] = max((c for t, c in temps if w["t0"] <= t <= w["t1"]), default=None)
        solo_ms = [w["cpu_ms"] for w in ws if w["kind"] == "solo" and w["cpu_ms"] and w["alive"]]
        if len(solo_ms) >= 2:
            noise.append((u["dir"], (max(solo_ms) - min(solo_ms)) / statistics.mean(solo_ms)))
        pos = 0
        for i, w in enumerate(ws):
            if w["kind"] != "corun":
                continue
            pos += 1
            refs = [x for x in (ws[i - 1] if i > 0 else None, ws[i + 1] if i + 1 < len(ws) else None)
                    if x and x["kind"] == "solo" and x["alive"] and x["cpu_ms"]]
            cs = mean([x["cpu_ms"] for x in refs])
            gs = mean(gsolo.get((u["trial"], w["gpu_mhz"]), [])) or mean([v for (t, g), vs in gsolo.items() if g == w["gpu_mhz"] for v in vs])
            gc = 1000.0 / w["gpu"]["avg_ts"]
            note = [] if w["alive"] else ["CPU process ended inside this window"]
            if len(refs) < 2:
                note.append(f"{len(refs)} solo reference(s)")
            rows.append({"trial": u["trial"], "mode": u["mode"], "a720": mlabel(u["a720"]), "gpu_mhz": w["gpu_mhz"], "gpu_pos": pos,
                         "cpu_solo_ms": cs, "cpu_corun_ms": w["cpu_ms"], "cpu_slowdown": (w["cpu_ms"] / cs - 1) if cs and w["cpu_ms"] else None,
                         "cpu_tokens": w["cpu_n"], "gpu_solo_ms": gs, "gpu_corun_ms": gc, "gpu_slowdown": (gc / gs - 1) if gs else None,
                         "a720_GBps_solo": mean([x["bw"] for x in refs]), "a720_GBps_corun": w["bw"],
                         "gpu_est_GBps": w["gpu"]["model_size"] * w["gpu"]["avg_ts"] / 1e9, "temp_max_c": w["tmax"],
                         "a720_ok": u.get("a720_ok"), "gpu_ok": w.get("gpu_ok"), "note": "; ".join(note), "_valid": w["alive"] and bool(refs)})
    with open(out_dir / "windows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=WIN_COLS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: S.fmt(r.get(k)) for k in WIN_COLS})

    good = [r for r in rows if r["_valid"] and r["a720_ok"] is not False and r["gpu_ok"]]
    groups = {}
    for r in good:
        groups.setdefault((r["mode"], r["a720"], r["gpu_mhz"]), []).append(r)
    summ = []
    for (m, c, g), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        row = {"mode": m, "a720": c, "gpu_mhz": g, "n": len(rs)}
        for k in SUM_COLS[4:]:
            row[k] = mean([r[k] for r in rs])
        summ.append(row)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=SUM_COLS)
        wr.writeheader()
        for r in summ:
            wr.writerow({k: S.fmt(r.get(k)) for k in SUM_COLS})

    f2 = lambda v: "-" if v is None else f"{v:.1f}"                     # noqa: E731
    pc = lambda v: "-" if v is None else f"{100 * v:+.1f}%"             # noqa: E731
    print(f"\nsummary ({len(good)} of {len(rows)} co-run windows usable; mean over trials). slowdown = co-run / solo - 1")
    print(f"{'A720':>5s} {'GPU':>5s} {'mode':>5s} {'n':>2s} | {'cpu solo':>8s} {'corun':>7s} {'slow':>7s} | {'gpu solo':>8s} {'corun':>7s} {'slow':>7s} | "
          f"{'A720 GB/s solo':>14s} {'corun':>6s}")
    for r in summ:
        print(f"{r['a720']:>5s} {r['gpu_mhz']:>5d} {r['mode']:>5s} {r['n']:>2d} | {f2(r['cpu_solo_ms']):>8s} {f2(r['cpu_corun_ms']):>7s} "
              f"{pc(r['cpu_slowdown']):>7s} | {f2(r['gpu_solo_ms']):>8s} {f2(r['gpu_corun_ms']):>7s} {pc(r['gpu_slowdown']):>7s} | "
              f"{f2(r['a720_GBps_solo']):>14s} {f2(r['a720_GBps_corun']):>6s}")
    by = {(r["mode"], r["a720"], r["gpu_mhz"]): r for r in summ}
    pairs = sorted({(c, g) for _, c, g in by})
    if any(("ram", c, g) in by and ("swap", c, g) in by for c, g in pairs):
        print("\nKV swap vs RAM KV (the question): GPU slowdown and A720 bandwidth during the co-run")
        for c, g in pairs:
            ram, swp = by.get(("ram", c, g)), by.get(("swap", c, g))
            if ram and swp and ram["gpu_slowdown"] is not None and swp["gpu_slowdown"] is not None:
                print(f"  A720 {c:>4s} GPU {g:>4d}: GPU slowdown ram {pc(ram['gpu_slowdown'])} -> swap {pc(swp['gpu_slowdown'])}"
                      f"  ({100 * (swp['gpu_slowdown'] - ram['gpu_slowdown']):+.1f} points);  A720 GB/s ram {f2(ram['a720_GBps_corun'])} -> swap {f2(swp['a720_GBps_corun'])}")
    if noise:
        worst = max(noise, key=lambda x: x[1])
        print(f"\nnoise floor: solo windows inside one CPU process differ by up to {100 * worst[1]:.1f}% ({worst[0]}); "
              f"median {100 * statistics.median(x for _, x in noise):.1f}%. A CPU slowdown smaller than this is not a result.")
    for tag, n_reads, n_tok in checks:
        flag = "OK" if abs(n_reads - n_tok) <= max(2, 0.01 * n_tok) else "MISMATCH: reads are not tokens, use the byte rate instead"
        print(f"token check {tag}: {n_reads} reads vs {n_tok} generated tokens -> {flag}")
    if len(good) < len(rows):
        print(f"WARNING: {len(rows) - len(good)} co-run window(s) left out (CPU process ended, no solo reference, or a frequency check failed): see windows.csv")
    print("results:", out_dir)


# ------------------------------------------------------------------ main
def main():
    a = build_parser().parse_args()
    if a.analyze:
        analyze(Path(a.analyze))
        return
    modes = a.modes.split(",")
    for m in modes:
        if m not in ("ram", "swap"):
            sys.exit(f"unknown mode {m!r}: use ram and/or swap")
    a720s, gfs = parse_a720(a.a720), [int(x) for x in a.gpu_freqs.split(",")]
    for c in a720s:
        if c is not None and not 800 <= c <= 2200:
            sys.exit(f"A720 {c} MHz is outside 800..2200")
    a.cpu_ids = cpu_ids(a.cpus)
    sched = est_schedule_s(a, gfs)
    if a.ntok == 0:
        a.ntok = int(math.ceil(sched * 1.3 / (a.min_cpu_ms / 1000.0)))
    p_est = int(a.prompt_chars / a.chars_per_token)
    if p_est + a.ntok + 256 > a.ctx:
        sys.exit(f"--ctx {a.ctx} is too small for about {p_est} prompt + {a.ntok} generated tokens: raise --ctx or lower --ntok")
    per_cpu = {c: est_prefill_s(a, c) + (a.ntok * 0.07 if a.let_finish else sched) + 60 for c in a720s}
    total = a.trials * (len(modes) * sum(per_cpu.values()) + len(gfs) * (est_gpu_job_s(a, 1000) + 30))
    print(f"plan: modes {modes} x A720 {[mlabel(c) for c in a720s]} -> {len(modes) * len(a720s)} CPU process(es) + {len(gfs)} solo_gpu unit(s) per trial, "
          f"{a.trials} trial(s); GPU {gfs} MHz inside each CPU process\n"
          f"      prompt {a.prompt_chars} chars (~{p_est} tokens), ctx {a.ctx}, up to {a.ntok} generated tokens, A720 {a.cpus} x{a.threads}; "
          f"schedule ~{sched:.0f} s after the first token\n"
          f"      estimate: " + ", ".join(f"A720 {mlabel(c)} ~{v / 60:.0f} min per CPU process" for c, v in per_cpu.items())
          + f" -> about {total / 3600:.1f} h in all (prefill rate from thermal-ab-20261113; rough)")
    if a.dry_run:
        return
    if platform.machine() != "aarch64":
        sys.exit("run this on the board")
    need = [a.model, a.llama_bin, a.run_stage1, a.gpu_bench, a.prompt_src] + ([a.adapter] if a.adapter and "swap" in modes else [])
    for p in need:
        if not Path(p).exists():
            sys.exit(f"missing: {p}")
    if subprocess.run(["sudo", "-n", "true"]).returncode:
        sys.exit("passwordless `sudo -n true` is needed (frequency pinning, run_stage1.sh drops the page cache with it)")

    done = set()
    if a.resume:
        out_dir = Path(a.resume)
        old = json.loads((out_dir / "meta.json").read_text())
        for k in KEY_ARGS:
            if old["args"].get(k) != vars(a).get(k):
                sys.exit(f"--resume: flag {k} differs from the original run ({old['args'].get(k)!r} vs {vars(a).get(k)!r})")
        if (out_dir / "units.jsonl").exists():
            for line in open(out_dir / "units.jsonl"):
                u = json.loads(line)
                if "error" not in u:
                    done.add((u["unit"], u["trial"], u.get("mode"), mlabel(u.get("a720")), u.get("gpu_mhz") if u["unit"] == "solo_gpu" else None))
        old.setdefault("resumed_at", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
        (out_dir / "meta.json").write_text(json.dumps(old, indent=1))
        print(f"resuming {out_dir}: {len(done)} finished unit(s) kept", flush=True)
    else:
        out_dir = Path(a.out_dir) if a.out_dir else REPO / "runs" / time.strftime("corunkv_%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt.txt").write_text(Path(a.prompt_src).read_text(errors="ignore")[:a.prompt_chars])
        (out_dir / "meta.json").write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "kernel": platform.release(),
                                                       "temp_start_c": S.temp_c()}, indent=1))
    prompt = out_dir / "prompt.txt"

    def _on_signal(sig, _frame):                       # log who stopped us, then unwind through the finally blocks (frequencies restored)
        print(f"received signal {signal.Signals(sig).name}: stopping", flush=True)
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)       # a dropped ssh session must not stop a long run
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    fq, host = S.Freq(), C.HostPin(a.host_cpu)
    try:
        host.pin(a.host_mhz * 1000)
        for trial in range(1, a.trials + 1):
            rng = random.Random(trial)
            order = [("solo_gpu", None, None, g) for g in gfs] + [("cpu", m, c, None) for m in modes for c in a720s]
            rng.shuffle(order)
            for kind, m, c, g in order:
                if (kind, trial, m, mlabel(c), g) in done:
                    continue
                if kind == "solo_gpu":
                    u = run_gpu_unit(trial, g, a, fq)
                else:
                    u = run_cpu_unit(trial, m, c, rng.sample(gfs, len(gfs)), a, fq, out_dir, prompt)
                with open(out_dir / "units.jsonl", "a") as f:
                    f.write(json.dumps(u) + "\n")
    except KeyboardInterrupt:
        print("interrupted; restoring frequencies. Partial results are in", out_dir)
    finally:
        fq.restore()
        host.restore()
    if (out_dir / "units.jsonl").exists():
        analyze(out_dir)


if __name__ == "__main__":
    main()
