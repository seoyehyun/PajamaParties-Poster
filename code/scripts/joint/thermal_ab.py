#!/usr/bin/env python3
"""thermal_ab.py -- does KV swap + CPU frequency control beat the NAIVE run on temperature, at equal work and bounded slowdown?

Hypothesis under test: with KV swap (KV cache on NVMe, selective reads, reuse buffer) and a pinned lower A720 frequency, the same
request (same prompt, same number of generated tokens) heats the board less than the naive run -- without lowering quality and
without slowing decode beyond --tpot-slack.

  naive        KV in RAM (full attention), default governors (schedutil), nothing pinned        <- the baseline
  freq_<MHz>   KV in RAM, the four A720 policies pinned to <MHz>                                (frequency control alone)
  kv_default   KV swap, default governors                                                       (KV swap alone)
  kv_<MHz>     KV swap + pinned A720 frequency                                                  (both)
  gnaive       GPU scenario baseline: Vulkan build, -ngl 99, ONE host thread on an A520, default governors/devfreq (no KV swap: see below)
  g<GPU>_c<A720|def>  GPU scenario: GPU devfreq pinned to <GPU> MHz for the whole request, A720 policies pinned to <A720> MHz (or left
                 on the default governor with "def"); e.g. g600_c800, g1000_cdef. The A720 cores are idle (the host thread runs on cpu2).
                 KV swap is NOT combined with the GPU path: with -ngl 99 the KV buffer is on the GPU and the patch crashes (exit 139);
                 with -nkvo it runs but 6.6x slower prefill / 12x slower decode (docs/joint/gpu-axis-experiment-plan.md, step 0).
  p<N>_g<GPU|def>_c<A720|def>   PARTIAL OFFLOAD: Vulkan build, -ngl N (N of 28 layers on the GPU, the rest on the A720 cores, 8 threads),
                 GPU devfreq pinned to <GPU> MHz and the A720 policies to <A720> MHz (def = default governor); p<N>_gdef_cdef is the
                 baseline of its N (docs/joint/partial-offload-experiment-plan.md). KV in RAM, no KV swap.
  co_g<GPU|def>_c<A720|def>   CO-RUN (two separate processes at the same time): a GPU job (Vulkan build, -ngl 99, one host thread on an A520)
                 and --co-cpu-reqs back-to-back CPU requests (CPU build, A720 x8 threads), same prompt and --ntok for every request.
                 GPU devfreq pinned to <GPU> MHz, A720 policies to <A720> MHz (def = default governors). co_gdef_cdef is the baseline.
                 tpot/ttft columns = the GPU job; the CPU job's tpot is reported and gated too (docs/joint/gpu-axis-experiment-plan.md, S3).
  ck_<ram|swap>_g<GPU|def>_c<A720|def>   CO-RUN WITH THE CPU JOB'S KV PLACEMENT AS A VARIABLE: one CPU request (run_stage1.sh, KV in RAM or KV swap, long prompt)
                 and, from its first generated token on, one GPU job (Vulkan build, -ngl 99, one host thread on an A520, short prompt, --ck-gpu-ntok tokens).
                 Same work in every config. tpot/ttft columns = the CPU job; the GPU job's tpot is reported and gated too (gpu_tpot_ms).
                 Heat is also integrated from the first generated token on (heat_post), which removes the prefill that is common to every config.
                 ck_ram_gdef_cdef is the baseline (docs/joint/corun-kvswap-plan.md, follow-up).
                 ck_<ram|swap>_g<GPU|def>_d<A720>: same, but the A720 is pinned only from the CPU job's first generated token on (like phase_).
                 ck_<ram|swap>_gov: same scenario, but nothing is pinned up front: governor_lite.py senses the state every 0.5 s (CPU prefill /
                 decode, GPU on / off) and applies the pre-computed table at run time; its decisions go to gov.csv / gov_events.jsonl.
                 --ck-gpu-delay N starts the GPU job N s after the CPU job's first token (with a smaller --ck-gpu-ntok it also ends before
                 the CPU job), so the state changes several times: prefill -> CPU decode -> co-run -> CPU decode. Use the same flags for every
                 config of the comparison; scripts/joint/gov_replay.py animates a gov run against the baseline.
                 --ck-gpu-bursts N (N >= 1): instead of one GPU job, N GPU jobs of --ck-gpu-ntok tokens each, --ck-gpu-off seconds apart
                 (fixed work, so every config does the same work). The CPU job's output size is sampled every 0.5 s (cpu_progress.csv) and
                 its speed is reported separately during the GPU bursts (cpu_Bps_on) and between them (cpu_Bps_off), in bytes of output
                 per second (a token-rate proxy; compare it between configs, not as tokens/s).
  phase_<MHz>    KV in RAM; prefill on the default governors, the A720 policies are pinned to <MHz> from the FIRST GENERATED TOKEN on
  kvphase_<MHz>  KV swap;   prefill on the default governors, pinned to <MHz> from the first generated token on
                 (phase-aware: keeps the time to first token, lowers the decode frequency. The first token is detected by out.txt
                  getting its first byte; llama-completion writes tokens as they are produced. switch_error_s checks that against
                  llama's own timing.)
Run order is interleaved over --repeat rounds. Each run = prefill + decode of the same prompt, then a COOL-DOWN TAIL that is part of
the measurement (it also serves as the cool-down before the next run). The baseline window and the tail always run on the DEFAULT
governors: a frequency is pinned only while the request runs (freq_/kv_<MHz>: from the start of the run; phase_: from the first token).

Metrics (per thermal zone i, dT_i(t) = T_i(t) - baseline_i, baseline = mean of the --pre seconds before the run):
  heat_area  = integral of dT_i dt from the start of the run until the board is back near baseline (tail). For a linear thermal
               system the integral of the temperature rise equals R_dc x (energy put into the heat source), whatever the time
               constants are. So heat_area per generated token is a RELATIVE proxy of energy per token INCLUDING prefill, KV writes
               and the CPU share of the I/O. It is not joules: the thermal resistance is unknown, the zones are all "acpitz" with an
               unknown mapping (the zone with the largest area in the naive run is used as the "hot zone"), readings are 1 C steps,
               NVMe heat is not visible (no NVMe hwmon), and leakage makes the system slightly non-linear.
  dT_ss      = mean dT over the last --window seconds of decode (steady-state proxy of average power during decode).
  tpot_ms    = decode ms per token (llama.cpp's own timing); prefill_s likewise.
  utilisation (scripts/sweep/util_sampler.py, 1 Hz), separately over the prefill (suffix _p) and the decode (suffix _d) of the main job
               (decode start = the detected first token for phase_/kvphase_/ck configs, else end of run - llama's decode time):
               cpu_util_a720 / cpu_iowait_a720 (busy / iowait share of the --cpus cores), cpu_util_gpu_host, cpu_util_all,
               disk_util_kv (io_ticks share of --dev) / disk_util_sda, disk_rd_MBps / disk_wr_MBps, gpu_busy (Mali runtime-PM active share),
               bus_GBps_a720 / bus_GBps_a520 (perf bus_access x --bus-bytes, CPU-side traffic only; only with perf_event_paranoid <= 0 or
               SUDO_PASS, else omitted). Why: a high disk_util alone does not say whether the disk is the bottleneck; with KV swap compare
               disk_util_kv and cpu_iowait_a720 (cores waiting on the reads) against cpu_util_a720 and bus_GBps (compute / DRAM bound).
Verdict per config vs naive: IMPROVED needs (a) heat_area/token lower than naive by more than the noise margin, (b) tpot <= naive x
--tpot-slack, (c) quality >= --min-quality, (d) prefill time (the time-to-first-token proxy) <= naive x --ttft-slack. Quality is an INPUT here (attention mass kept by the KV selection, measured earlier:
docs/kvswap/results/kvswap-8k.md, 4K, M=50, predictor r=128: 0.9448; full attention = 1.0); this script does not measure quality.

Needs: the board, ~/llama-base/build/bin/llama-completion (KV swap patches 01-05), scripts/kvswap/run_stage1.sh, the prompt and
adapter files. No sudo password: it only uses the NOPASSWD sudo rules (tee on cpufreq/devfreq/drop_caches, true).
  python3 scripts/joint/thermal_ab.py --dry-run
  python3 scripts/joint/thermal_ab.py                                   # 6 configs, 1 round (about 1.5 h)
  python3 scripts/joint/thermal_ab.py --configs naive,kv_1500 --repeat 2
  python3 scripts/joint/thermal_ab.py --analyze runs/thermal_<ts>       # recompute the report from saved runs
  python3 scripts/joint/thermal_ab.py --resume runs/thermal_<ts> <same flags as the original run>   # after a reboot: skips finished runs
Frequencies are restored on exit / Ctrl-C / SIGTERM. Raw output: runs/thermal_<timestamp>/ (git-ignored).
"""
import argparse
import csv
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # scripts/joint/
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "sweep"))            # sweep_cpufreq (helpers) lives in scripts/sweep/
import sweep_cpufreq as S   # noqa: E402
import util_sampler as U    # noqa: E402
import governor_lite as GL  # noqa: E402  (scripts/joint/, same folder)
UTIL_BASE = ["cpu_util_a720", "cpu_iowait_a720", "cpu_util_gpu_host", "cpu_util_all", "disk_util_kv", "disk_util_sda", "disk_rd_MBps", "disk_wr_MBps",
             "gpu_busy", "read_GBps_a720", "bus_GBps_a720", "bus_GBps_a520"]
UTIL_KEYS = [k + s for s in ("_p", "_d") for k in UTIL_BASE]
UTIL_SHOW = ["cpu_util_a720_d", "cpu_iowait_a720_d", "disk_util_kv_d", "disk_rd_MBps_d", "gpu_busy_d", "read_GBps_a720_d"]
ZONES = sorted(Path("/sys/class/thermal").glob("thermal_zone*"), key=lambda p: int(p.name.replace("thermal_zone", "")))
CDEVS = sorted(Path("/sys/class/thermal").glob("cooling_device*"), key=lambda p: int(p.name.replace("cooling_device", "")))


def parse_gpu(name):
    """gnaive | g<GPU MHz>_c<A720 MHz|def>  ->  (gpu_mhz or None, a720_mhz or None), or None if not a GPU config"""
    if name == "gnaive":
        return None, None
    m = re.fullmatch(r"g(\d+)_c(\d+|def)", name)
    if not m:
        return False
    return int(m.group(1)), (None if m.group(2) == "def" else int(m.group(2)))


def is_gpu(name):
    return name == "gnaive" or parse_gpu(name) is not False


def parse_part(name):
    """p<N>_g<GPU|def>_c<A720|def>  ->  (n_gpu_layers, gpu_mhz or None, a720_mhz or None), or None if not a partial-offload config"""
    m = re.fullmatch(r"p(\d+)_g(\d+|def)_c(\d+|def)", name)
    if not m:
        return None
    return int(m.group(1)), (None if m.group(2) == "def" else int(m.group(2))), (None if m.group(3) == "def" else int(m.group(3)))


def parse_co(name):
    """co_g<GPU|def>_c<A720|def>  ->  (gpu_mhz or None, a720_mhz or None), or None if not a co-run config"""
    m = re.fullmatch(r"co_g(\d+|def)_c(\d+|def)", name)
    if not m:
        return None
    return (None if m.group(1) == "def" else int(m.group(1))), (None if m.group(2) == "def" else int(m.group(2)))


def parse_ck(name):
    """ck_<ram|swap>_g<GPU|def>_c<A720|def>  ->  (kv, gpu_mhz or None, a720_mhz or None, phased), or None if not a ck config.
    ck_<ram|swap>_g<GPU|def>_d<A720>: the A720 is pinned only from the CPU job's first generated token on (prefill on the default governors)."""
    g = re.fullmatch(r"ck_(ram|swap)_gov", name)
    if g:
        return g.group(1), None, None, False
    m = re.fullmatch(r"ck_(ram|swap)_g(\d+|def)_(c(\d+|def)|d(\d+))", name)
    if not m:
        return None
    a720 = m.group(5) or m.group(4)
    return m.group(1), (None if m.group(2) == "def" else int(m.group(2))), (None if a720 == "def" else int(a720)), bool(m.group(5))


def is_gov(name):
    return bool(re.fullmatch(r"ck_(ram|swap)_gov", name))


def is_base(name):
    return name in ("naive", "gnaive", "co_gdef_cdef", "ck_ram_gdef_cdef") or bool(re.fullmatch(r"p\d+_gdef_cdef", name))


def parse_config(name):
    """naive | freq_<MHz> | kv_default | kv_<MHz> | phase_<MHz> | kvphase_<MHz> | gnaive | g<GPU>_c<A720|def>  ->  (kv, mhz, phased)"""
    if name == "naive":
        return "ram", None, False
    if is_gpu(name):
        return "ram", parse_gpu(name)[1], False        # mhz = the A720 frequency; the GPU frequency comes from parse_gpu
    if parse_part(name):
        return "ram", parse_part(name)[2], False
    if parse_co(name):
        return "ram", parse_co(name)[1], False
    if parse_ck(name):
        return parse_ck(name)[0], parse_ck(name)[2], parse_ck(name)[3]
    if name == "kv_default":
        return "swap", None, False
    m = re.fullmatch(r"(freq|kv|phase|kvphase)_(\d+)", name)
    if not m:
        raise SystemExit(f"unknown config {name!r}: use naive, freq_<MHz>, kv_default, kv_<MHz>, phase_<MHz> or kvphase_<MHz>")
    kind = m.group(1)
    return ("ram" if kind in ("freq", "phase") else "swap"), int(m.group(2)), kind in ("phase", "kvphase")


def build_parser():
    h = Path.home()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="naive,freq_1500,freq_1200,kv_default,kv_1500,kv_1200")
    ap.add_argument("--repeat", type=int, default=1, help="rounds; each round runs every config once, in a rotated order")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ntok", type=int, default=3000, help="generated tokens (the same for every config; decode must outlast --window)")
    ap.add_argument("--prompt-src", default="/mnt/ssd/prompts/haystack_c4.txt")
    ap.add_argument("--prompt-chars", type=int, default=16000, help="first N characters (16000 = about 3.5K tokens)")
    ap.add_argument("--model", default=str(h / "PajamaParties/models/Qwen3-1.7B-Q4_K_M.gguf"))
    ap.add_argument("--llama-bin", default=str(h / "llama-base/build/bin/llama-completion"))
    ap.add_argument("--run-stage1", default=str(HERE.parent / "kvswap" / "run_stage1.sh"))
    ap.add_argument("--gpu-bin", default=str(h / "llama-base/build-vulkan/bin/llama-completion"), help="Vulkan build used by the GPU configs")
    ap.add_argument("--gpu-host-cpu", default="2", help="core of the single host thread of the GPU configs (an A520)")
    ap.add_argument("--co-cpu-reqs", type=int, default=4, help="co-run configs: number of back-to-back CPU requests that run beside the one GPU request")
    ap.add_argument("--ck-gpu-ntok", type=int, default=3000, help="ck configs: tokens the GPU job generates (should roughly cover the CPU job's decode)")
    ap.add_argument("--ck-gpu-prompt-chars", type=int, default=450, help="ck configs: first N characters of the prompt for the (short) GPU job")
    ap.add_argument("--ck-gpu-ctx", type=int, default=4096)
    ap.add_argument("--ck-gpu-delay", type=float, default=0.0, help="ck configs: start the GPU job this many seconds after the CPU job's first token")
    ap.add_argument("--ck-gpu-bursts", type=int, default=0, help="ck configs: N GPU jobs of --ck-gpu-ntok tokens, --ck-gpu-off s apart (0 = one job, the old behaviour)")
    ap.add_argument("--ck-gpu-off", type=float, default=45.0, help="ck configs with --ck-gpu-bursts: idle seconds between two GPU jobs")
    ap.add_argument("--gov-period", type=float, default=0.5, help="ck_*_gov: governor poll period (s)")
    ap.add_argument("--gov-stable", type=int, default=2, help="ck_*_gov: polls a new state must persist before it is applied")
    ap.add_argument("--cpus", default="0-1,6-11", help="A720 cores (taskset list)")
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
    ap.add_argument("--window", type=float, default=60.0, help="seconds at the end of decode for dT_ss")
    ap.add_argument("--pre", type=float, default=45.0, help="idle seconds before each run (baseline)")
    ap.add_argument("--tail-max", type=float, default=900.0, help="longest cool-down tail per run")
    ap.add_argument("--eps", type=float, default=1.0, help="tail ends when every zone is within this many C of baseline for 20 s")
    ap.add_argument("--tpot-slack", type=float, default=1.10, help="allowed tpot vs naive")
    ap.add_argument("--ttft-slack", type=float, default=1.10, help="allowed prefill time (time-to-first-token proxy) vs naive")
    ap.add_argument("--min-quality", type=float, default=0.93)
    ap.add_argument("--quality-swap", type=float, default=0.9448, help="quality score assigned to KV-swap configs (input, see header)")
    ap.add_argument("--min-gain", type=float, default=0.05, help="smallest heat_area/token reduction that counts (also >= 2x naive spread)")
    ap.add_argument("--perf", default=U.DEFAULT_PERF, help="perf binary for the bus_access counts (bus_GBps_*)")
    ap.add_argument("--bus-bytes", type=float, default=31.5, help="bytes per bus_access event (A720 calibration, membw_calib)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--analyze", default=None, help="recompute the report of an existing run directory")
    ap.add_argument("--resume", default=None, help="continue an interrupted run directory (same flags as before): finished (config, round) pairs are skipped")
    ap.add_argument("--only", default=None, help="with --resume: run only these configs (comma list) of the remaining ones, e.g. the RAM-KV ones first")
    ap.add_argument("--dry-run", action="store_true")
    return ap


# ------------------------------------------------------------------ sampling
def sample():
    return {"t": time.time(), "T": [int(S.rd(z / "temp")) / 1000.0 for z in ZONES], "cd": max(int(S.rd(c / "cur_state")) for c in CDEVS),
            "cpu6": int(S.rd(S.CPUFREQ / "policy6/cpuinfo_cur_freq")), "gpu": int(S.rd(S.GPU_DEV / "cur_freq"))}


class Trace:
    def __init__(self):
        self.rows, self.stop, self.th = [], threading.Event(), None

    def start(self):
        def loop():
            while not self.stop.is_set():
                self.rows.append(sample())
                self.stop.wait(2.0)
        self.th = threading.Thread(target=loop, daemon=True)
        self.th.start()

    def end(self):
        self.stop.set()
        self.th.join(timeout=10)


class CoProc:
    """The two jobs of a co-run config. Looks like a Popen: poll() / wait() give the worst exit code of both."""
    def __init__(self, a, rdir, prompt):
        def cmd(binp, cpus, thr, extra):
            return ["taskset", "-c", cpus, binp, "-m", a.model, "-f", str(prompt), "-c", str(a.ctx), "-n", str(a.ntok), "--ignore-eos",
                    "--temp", "0", "--seed", "1", "-fa", "on", "-t", str(thr), "--no-warmup", "--no-display-prompt", "-lv", "4"] + extra
        self.rcs = []
        self.t_gpu_end = None                          # the GPU job ends before the CPU requests: its decode window ends here
        self.gpu = subprocess.Popen(cmd(a.gpu_bin, a.gpu_host_cpu, 1, ["-ngl", "99"]), stdin=subprocess.DEVNULL,
                                    stdout=open(rdir / "gpu_out.txt", "w"), stderr=open(rdir / "gpu_err.txt", "w"))

        def gpu_watch():
            self.gpu.wait()
            self.t_gpu_end = time.time()
        threading.Thread(target=gpu_watch, daemon=True).start()

        def cpu_loop():
            for i in range(a.co_cpu_reqs):
                r = subprocess.run(cmd(a.llama_bin, a.cpus, a.threads, []), stdin=subprocess.DEVNULL,
                                   stdout=open(rdir / f"cpu{i}_out.txt", "w"), stderr=open(rdir / f"cpu{i}_err.txt", "w"))
                self.rcs.append(r.returncode)
        self.th = threading.Thread(target=cpu_loop)
        self.th.start()

    def poll(self):
        g = self.gpu.poll()
        if g is None or self.th.is_alive():
            return None
        return max([g] + self.rcs)

    def wait(self):
        g = self.gpu.wait()
        self.th.join()
        return max([g] + self.rcs)


class CoKV:
    """ck config: the CPU job is run_stage1.sh; from its first token on (+ --ck-gpu-delay), one GPU job, or --ck-gpu-bursts GPU jobs
    --ck-gpu-off s apart. The CPU job's output size is sampled every 0.5 s (progress). poll() / wait() give the worst exit code."""
    def __init__(self, a, rdir, env, mode, gpu_prompt, logf):
        self.t_first, self.gpu, self.t_gpu_start, self.t_gpu_end = None, None, None, None
        self.bursts, self.progress = [], []            # bursts: (start, end, rc, file tag); progress: (time, bytes of out.txt)
        self.cpu = subprocess.Popen(["bash", a.run_stage1, mode], env=env, stdout=logf, stderr=subprocess.STDOUT)
        outp = rdir / "out.txt"

        def sampler():
            while self.cpu.poll() is None:
                try:
                    self.progress.append((time.time(), outp.stat().st_size if outp.exists() else 0))
                except OSError:
                    pass
                time.sleep(0.5)
            try:
                self.progress.append((time.time(), outp.stat().st_size))
            except OSError:
                pass
        self.sth = threading.Thread(target=sampler, daemon=True)
        self.sth.start()

        def gpu_cmd():
            return ["taskset", "-c", a.gpu_host_cpu, a.gpu_bin, "-m", a.model, "-f", str(gpu_prompt), "-c", str(a.ck_gpu_ctx),
                    "-n", str(a.ck_gpu_ntok), "--ignore-eos", "--temp", "0", "--seed", "1", "-fa", "on", "-t", "1", "--no-warmup",
                    "--no-display-prompt", "-lv", "4", "-ngl", "99"]

        def wait_until(t):
            while time.time() < t:
                if self.cpu.poll() is not None:
                    return False
                time.sleep(0.1)
            return True

        def starter():
            while self.cpu.poll() is None:
                try:
                    if outp.exists() and outp.stat().st_size > 0:
                        break
                except OSError:
                    pass
                time.sleep(0.1)
            else:
                return                                 # the CPU job ended before its first token: no GPU job
            self.t_first = time.time()
            if not wait_until(self.t_first + a.ck_gpu_delay):
                return                                 # the CPU job ended before the delayed GPU start
            for i in range(max(1, a.ck_gpu_bursts)):
                if i and not wait_until(self.bursts[-1][1] + a.ck_gpu_off):
                    break                              # the CPU job ended: no more bursts
                tag = f"_b{i}" if a.ck_gpu_bursts else ""
                t_s = time.time()
                p = subprocess.Popen(gpu_cmd(), stdin=subprocess.DEVNULL, stdout=open(rdir / f"gpu{tag}_out.txt", "w"),
                                     stderr=open(rdir / f"gpu{tag}_err.txt", "w"))
                if i == 0:
                    self.gpu, self.t_gpu_start = p, t_s
                rc = p.wait()
                self.bursts.append((t_s, time.time(), rc, tag))
                self.t_gpu_end = self.bursts[-1][1]
        self.th = threading.Thread(target=starter)
        self.th.start()

    def _gpu_rc(self):
        return max(b[2] for b in self.bursts) if self.bursts else 1

    def poll(self):
        if self.th.is_alive():
            return None
        c = self.cpu.poll()
        if c is None:
            return None
        return max(c, self._gpu_rc())

    def wait(self):
        self.th.join()
        c = self.cpu.wait()
        return max(c, self._gpu_rc())

    def segment_rates(self, t_dec_start, t_dec_end):
        """CPU job output rate (bytes/s) inside and outside the GPU bursts, over its decode."""
        on = {"b": 0.0, "s": 0.0}
        off = {"b": 0.0, "s": 0.0}
        for (t0_, b0), (t1_, b1) in zip(self.progress, self.progress[1:]):
            mid = (t0_ + t1_) / 2
            if not (t_dec_start <= mid <= t_dec_end) or b1 < b0:
                continue
            seg = on if any(bs <= mid <= be for bs, be, _, _ in self.bursts) else off
            seg["b"] += b1 - b0
            seg["s"] += t1_ - t0_
        return {"cpu_Bps_on": on["b"] / on["s"] if on["s"] else None, "cpu_on_s": on["s"],
                "cpu_Bps_off": off["b"] / off["s"] if off["s"] else None, "cpu_off_s": off["s"]}


# ------------------------------------------------------------------ one run
def run_one(name, rep, a, fq, out_dir, prompt):
    kv, mhz, phased = parse_config(name)
    rdir = out_dir / f"{name}_r{rep}"
    rdir.mkdir(parents=True, exist_ok=True)
    for stale in ["out.txt", "err.txt", "meta.txt", "iostat.log", "mpstat.log", "trace.csv", "run_stage1.log", "gpu_out.txt", "gpu_err.txt"] + \
            [f"cpu{i}_{k}.txt" for i in range(20) for k in ("out", "err")] + [f"gpu_b{i}_{k}.txt" for i in range(20) for k in ("out", "err")] + ["cpu_progress.csv"]:
        (rdir / stale).unlink(missing_ok=True)        # a resumed run may find the partial files of the interrupted one
    fq.restore()                                   # every run starts from the default governors (baseline window and prefill)
    gpu = is_gpu(name)
    part = parse_part(name)
    co = parse_co(name)
    ck = parse_ck(name)
    gmhz = parse_gpu(name)[0] if gpu else (part[1] if part else (co[0] if co else (ck[1] if ck else None)))
    use_vk = gpu or bool(part)
    env = {**os.environ, "LLAMA_BIN": a.gpu_bin if use_vk else a.llama_bin, "MODEL": a.model, "PROMPT": str(prompt), "KV_FILE": a.kv_file, "CTX": str(a.ctx),
           "NTOK": str(a.ntok), "CPUS": a.gpu_host_cpu if gpu else a.cpus, "THREADS": "1" if gpu else str(a.threads), "DEV": a.dev, "OUTDIR": str(rdir)}
    if gpu:
        env["EXTRA"] = "-ngl 99"
    elif part:
        env["EXTRA"] = f"-ngl {part[0]}"
    if kv == "swap":
        env.update(M=str(a.m), G=str(a.g), W=str(a.w), SINK=str(a.sink), PREFETCH="1", KV_IO="direct", IO_THREADS=str(a.io_threads),
                   LLAMA_KV_REUSE=str(a.reuse))
        if a.adapter:
            env["ADAPTER"] = a.adapter
    mode = "base" if kv == "ram" else "select"

    tr = Trace()
    tr.start()
    ul = U.UtilLog({"a720": U.cpulist(a.cpus), "gpu_host": U.cpulist(a.gpu_host_cpu), "all": None}, {"kv": a.dev, "sda": "sda"},
                   perf_bin=a.perf, mode=getattr(a, "perf_mode", None), bus_bytes=a.bus_bytes, tmp_dir=rdir).start()
    time.sleep(a.pre)
    t0 = time.time()
    print(f"[{name} r{rep}] start  temp_max={max(tr.rows[-1]['T']):.0f}C  cpu6={tr.rows[-1]['cpu6'] // 1000} MHz", flush=True)
    if mhz is not None and not phased:
        fq.pin_cpu(mhz * 1000)                     # pinned for the whole request
    if gmhz is not None:
        fq.pin_gpu(gmhz * 1000000)
    t_switch, gov = None, None
    with open(rdir / "run_stage1.log", "w") as logf:
        if co:
            try:
                S.wr("/proc/sys/vm/drop_caches", 3)        # one cold start for both jobs (run_stage1.sh does this per run; here the jobs are launched directly)
            except RuntimeError:
                pass
            proc = CoProc(a, rdir, prompt)
        elif ck:
            gprompt = out_dir / "gpu_prompt.txt"
            if not gprompt.exists():
                gprompt.write_text(prompt.read_text(errors="ignore")[:a.ck_gpu_prompt_chars])
            proc = CoKV(a, rdir, env, mode, gprompt, logf)
            if is_gov(name):
                for stale in ("gov.csv", "gov_events.jsonl"):
                    (rdir / stale).unlink(missing_ok=True)
                gov = GL.Governor(GL.Actuator(fq), GL.Sensors(a.cpus, rdir / "out.txt"), kv, rdir / "gov.csv",
                                  period=a.gov_period, stable=a.gov_stable, t0=t0).start()
        else:
            proc = subprocess.Popen(["bash", a.run_stage1, mode], env=env, stdout=logf, stderr=subprocess.STDOUT)
        if phased:                                 # wait for the first generated token, then pin
            outp = rdir / "out.txt"
            while proc.poll() is None:
                try:
                    if outp.exists() and outp.stat().st_size > 0:
                        fq.pin_cpu(mhz * 1000)
                        t_switch = time.time()
                        break
                except OSError:
                    pass
                time.sleep(0.1)
        rc = proc.wait()
        if gov:
            gov.stop()
    t1 = time.time()
    ul.end()                                       # utilisation covers the request only, not the tail
    fq.restore()                                   # the cool-down tail runs on the default governors for every config
    base = [statistics.mean(s["T"][i] for s in tr.rows if s["t"] < t0) for i in range(len(ZONES))]
    ok_tail, quiet_since = False, None
    while time.time() - t1 < a.tail_max:
        s = tr.rows[-1]
        if max(s["T"][i] - base[i] for i in range(len(ZONES))) <= a.eps:
            quiet_since = quiet_since or s["t"]
            if s["t"] - quiet_since >= 20:
                ok_tail = True
                break
        else:
            quiet_since = None
        time.sleep(2)
    tr.end()

    errname = "gpu_err.txt" if co else "err.txt"
    err = (rdir / errname).read_text(errors="replace") if (rdir / errname).exists() else ""
    pe = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", err)
    ev = re.search(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs", err)
    res = {"config": name, "rep": rep, "kv": kv, "mhz": mhz, "gpu_mhz": gmhz, "gpu": gpu or bool(part) or bool(co) or bool(ck), "co": bool(co), "ck": bool(ck), "ngl": part[0] if part else None, "phased": phased, "rc": rc, "tail_complete": int(ok_tail),
           "zones": [z.name for z in ZONES], "t0": t0, "t1": t1, "wall_s": t1 - t0}
    if rc or not (pe and ev):
        res["error"] = "run failed or no timing in err.txt"
        print(f"   FAILED rc={rc}: see {rdir}/err.txt", flush=True)
        return res
    if phased:
        if t_switch is None:
            res["error"] = "decode start not detected: frequency was never pinned"
            print(f"   FAILED: {res['error']}", flush=True)
            return res
        res["t_switch"] = t_switch
        res["switch_error_s"] = t_switch - (t1 - float(ev.group(1)) / 1000)    # > 0: pinned after llama's own decode start
    res.update(prefill_s=float(pe.group(1)) / 1000, prefill_tokens=int(pe.group(2)), decode_s=float(ev.group(1)) / 1000,
               tokens=int(ev.group(2)), tpot_ms=float(ev.group(1)) / int(ev.group(2)))
    t_end = proc.t_gpu_end if co and getattr(proc, "t_gpu_end", None) else t1     # co: the GPU job's own end, not the last CPU request's
    t_dec = res.get("t_switch") or (proc.t_first if ck and getattr(proc, "t_first", None) else None) or (t_end - res["decode_s"])
    res["t_decode_start"] = t_dec
    res.update(ul.window(t_dec - res["prefill_s"], t_dec, "_p"))
    res.update(ul.window(t_dec, min(t_end, t_dec + res["decode_s"]), "_d"))
    if co:                                         # the CPU job: mean over its requests; "tokens" becomes the total of both jobs (heat per token of the whole work)
        cp, ct, tk = [], [], 0
        for i in range(a.co_cpu_reqs):
            ce = (rdir / f"cpu{i}_err.txt").read_text(errors="replace") if (rdir / f"cpu{i}_err.txt").exists() else ""
            cpe = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", ce)
            cev = re.search(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs", ce)
            if not (cpe and cev):
                res["error"] = f"CPU request {i} has no timing"
                print(f"   FAILED: {res['error']}", flush=True)
                return res
            cp.append(float(cpe.group(1)) / 1000)
            ct.append(float(cev.group(1)) / int(cev.group(2)))
            tk += int(cev.group(2))
        res.update(cpu_prefill_s=statistics.mean(cp), cpu_tpot_ms=statistics.mean(ct), cpu_tokens=tk, gpu_tokens=res["tokens"], tokens=res["tokens"] + tk)
    if ck:                                         # the GPU job(s): own timing; "tokens" becomes the total of both jobs
        g_ms, g_runs = 0.0, 0
        for tag in ([b[3] for b in proc.bursts] or [""]):
            ge = (rdir / f"gpu{tag}_err.txt").read_text(errors="replace") if (rdir / f"gpu{tag}_err.txt").exists() else ""
            gev = re.search(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs", ge)
            if not gev:
                res["error"] = f"GPU job{tag} has no timing"
                print(f"   FAILED: {res['error']}", flush=True)
                return res
            g_ms += float(gev.group(1)); g_runs += int(gev.group(2))
        res.update(gpu_tpot_ms=g_ms / g_runs, gpu_tokens=g_runs, cpu_tokens=res["tokens"],
                   tokens=res["tokens"] + g_runs, t_first=proc.t_first, t_gpu_start=proc.t_gpu_start, t_gpu_end=proc.t_gpu_end,
                   gpu_bursts=[[b[0], b[1]] for b in proc.bursts])
        res.update(proc.segment_rates(proc.t_first, t1))
        with open(rdir / "cpu_progress.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_rel_s", "out_bytes"])
            for t_, b_ in proc.progress:
                w.writerow([round(t_ - t0, 2), b_])
        if gov:
            res["gov_events"] = len(gov.events)
    # ---- thermal integrals
    rows = [s for s in tr.rows if s["t"] >= t0]
    A = [0.0] * len(ZONES)
    for p, q in zip(rows, rows[1:]):
        dt = q["t"] - p["t"]
        for i in range(len(ZONES)):
            A[i] += 0.5 * ((p["T"][i] - base[i]) + (q["T"][i] - base[i])) * dt
    if ck and proc.t_first:                        # heat from the first generated token on (the prefill is common to every config)
        prow = [s for s in tr.rows if s["t"] >= proc.t_first]
        Ap = [0.0] * len(ZONES)
        for p_, q_ in zip(prow, prow[1:]):
            dt = q_["t"] - p_["t"]
            for i in range(len(ZONES)):
                Ap[i] += 0.5 * ((p_["T"][i] - base[i]) + (q_["T"][i] - base[i])) * dt
        res["A_post"] = Ap
    win = a.window if res["decode_s"] >= a.window + 20 else res["decode_s"] / 2
    wrows = [s for s in rows if t1 - win <= s["t"] <= t1]
    res.update(A=A, base=base, window_s=win, window_ok=int(res["decode_s"] >= a.window + 20),
               dT_ss=[statistics.mean(s["T"][i] - base[i] for s in wrows) if wrows else None for i in range(len(ZONES))],
               peak_c=max(max(s["T"]) for s in rows), max_cooling_state=max(s["cd"] for s in rows),
               min_cpu6_mhz=min(s["cpu6"] for s in rows[2:]) // 1000 if len(rows) > 2 else None,
               gpu_mhz_mean=statistics.mean(s["gpu"] for s in rows) / 1e6, gpu_mhz_max=max(s["gpu"] for s in rows) / 1e6)
    with open(rdir / "trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_rel_s"] + [z.name for z in ZONES] + ["max_cooling_state", "cpu6_khz", "gpu_hz"])
        for s in tr.rows:
            w.writerow([round(s["t"] - t0, 1)] + s["T"] + [s["cd"], s["cpu6"], s["gpu"]])
    print(f"   prefill {res['prefill_s']:.0f} s, decode {res['tpot_ms']:.1f} ms/tok ({res['tokens']} tok), tail {'ok' if ok_tail else 'TRUNCATED'}, "
          f"peak {res['peak_c']:.0f}C, cooling state max {res['max_cooling_state']}"
          + (f", pinned {res['switch_error_s']:+.1f} s after llama's decode start" if phased else ""), flush=True)
    print("   decode util: " + "  ".join(f"{k[:-2]}={res[k]:.2f}" for k in UTIL_SHOW if k in res), flush=True)
    return res


# ------------------------------------------------------------------ report
def report(runs, a, out_dir):
    good = [r for r in runs if "A" in r]
    if not good:
        print("no successful runs")
        return
    names = list(dict.fromkeys(r["config"] for r in good))
    naive = [r for r in good if is_base(r["config"])]
    ref = naive or good
    nz = len(good[0]["A"])
    hot = max(range(nz), key=lambda i: statistics.mean(r["A"][i] for r in ref))
    zname = good[0]["zones"][hot]

    def agg(rs, f):
        v = [f(r) for r in rs]
        return statistics.mean(v), min(v), max(v)
    rows_out = []
    nv = None
    for n in names:
        rs = [r for r in good if r["config"] == n]
        tpot = agg(rs, lambda r: r["tpot_ms"])
        apt = agg(rs, lambda r: r["A"][hot] / r["tokens"])
        dss = agg(rs, lambda r: r["dT_ss"][hot] if r["dT_ss"][hot] is not None else float("nan"))
        mlabel = ("decode %d" % rs[0]["mhz"]) if rs[0].get("phased") else (rs[0]["mhz"] or "default")
        if rs[0].get("gpu"):
            mlabel = "gpu %s / a720 %s" % (rs[0].get("gpu_mhz") or "default", rs[0]["mhz"] or "default")
        row = {"config": n, "n": len(rs), "kv": rs[0]["kv"], "mhz": mlabel, "tpot_ms": tpot[0], "prefill_s": statistics.mean(r["prefill_s"] for r in rs),
               "total_s": statistics.mean(r["prefill_s"] + r["decode_s"] for r in rs),
               "switch_error_s": statistics.mean(r["switch_error_s"] for r in rs) if rs[0].get("phased") else None,
               "heat_area_per_tok": apt[0], "heat_area_min": apt[1], "heat_area_max": apt[2], "dT_ss": dss[0], "peak_c": max(r["peak_c"] for r in rs), "gpu_mhz_mean": statistics.mean(r.get("gpu_mhz_mean") or 0 for r in rs),
               "max_cooling_state": max(r["max_cooling_state"] for r in rs), "tail_complete": min(r["tail_complete"] for r in rs),
               "window_ok": min(r["window_ok"] for r in rs), "quality": 1.0 if rs[0]["kv"] == "ram" else a.quality_swap,
               "cpu_tpot_ms": statistics.mean(r["cpu_tpot_ms"] for r in rs) if rs[0].get("co") else None,
               "gpu_tpot_ms": statistics.mean(r["gpu_tpot_ms"] for r in rs) if rs[0].get("ck") else None,
               "heat_post_per_tok": statistics.mean(r["A_post"][hot] / r["tokens"] for r in rs) if rs[0].get("ck") and all("A_post" in r for r in rs) else None,
               **{k: statistics.mean(r[k] for r in rs) for k in ("cpu_Bps_on", "cpu_Bps_off") if all(r.get(k) for r in rs)},
               **{k: statistics.mean(r[k] for r in rs if k in r) for k in UTIL_KEYS if any(k in r for r in rs)}}
        rows_out.append(row)
        if is_base(n):
            nv = row
    print(f"\nreport (hot zone = {zname}: largest heat area in the {'naive' if naive else 'reference'} run; heat_area = C*s per generated token, "
          f"prefill + decode + cool-down tail)")
    hdr = f"{'config':14s} {'n':>2s} {'tpot ms':>8s} {'vs naive':>8s} {'ttft x':>7s} {'total x':>7s} {'dT_ss C':>8s} {'heat/tok':>9s} {'vs naive':>9s} {'peak C':>6s} {'qual':>6s}  verdict"
    print(hdr)
    margin = a.min_gain
    if nv and nv["n"] > 1:
        margin = max(margin, 2 * (nv["heat_area_max"] - nv["heat_area_min"]) / (2 * nv["heat_area_per_tok"]))
    for row in rows_out:
        vt = row["tpot_ms"] / nv["tpot_ms"] if nv else None
        vh = row["heat_area_per_tok"] / nv["heat_area_per_tok"] if nv else None
        vf = row["prefill_s"] / nv["prefill_s"] if nv else None
        vw = row["total_s"] / nv["total_s"] if nv else None
        if is_base(row["config"]):
            verdict = "baseline"
        elif not nv:
            verdict = "no naive run to compare"
        else:
            gain = vh < 1 - margin
            vc = (row["cpu_tpot_ms"] / nv["cpu_tpot_ms"]) if row.get("cpu_tpot_ms") and nv.get("cpu_tpot_ms") else None
            row["vs_naive_cpu_tpot"] = vc
            vg = (row["gpu_tpot_ms"] / nv["gpu_tpot_ms"]) if row.get("gpu_tpot_ms") and nv.get("gpu_tpot_ms") else None
            row["vs_naive_gpu_tpot"] = vg
            row["vs_naive_heat_post"] = (row["heat_post_per_tok"] / nv["heat_post_per_tok"]) if row.get("heat_post_per_tok") and nv.get("heat_post_per_tok") else None
            slow_ok = vt <= a.tpot_slack and (vc is None or vc <= a.tpot_slack) and (vg is None or vg <= a.tpot_slack)
            q_ok = row["quality"] >= a.min_quality
            ttft_ok = vf <= a.ttft_slack
            verdict = ("IMPROVED" if gain and slow_ok and q_ok and ttft_ok else
                       "heat lower but TOO SLOW" if gain and not slow_ok else
                       "heat lower but TTFT too long" if gain and slow_ok and q_ok and not ttft_ok else
                       "heat lower but QUALITY below gate" if gain and not q_ok else
                       "no clear gain (within noise)" if abs(vh - 1) <= margin else "WORSE (more heat)")
        row.update(vs_naive_tpot=vt, vs_naive_heat=vh, vs_naive_ttft=vf, vs_naive_total=vw, verdict=verdict)
        print(f"{row['config']:14s} {row['n']:>2d} {row['tpot_ms']:>8.1f} {('%.2fx' % vt) if vt else '-':>8s} {('%.2fx' % vf) if vf else '-':>7s} "
              f"{('%.2fx' % vw) if vw else '-':>7s} {row['dT_ss']:>8.1f} {row['heat_area_per_tok']:>9.2f} "
              f"{('%.2fx' % vh) if vh else '-':>9s} {row['peak_c']:>6.0f} {row['quality']:>6.3f}  {verdict}")
    print(f"\nnoise margin for 'gain' = {margin:.0%} (max of --min-gain and 2x the naive spread when --repeat >= 2); tpot slack {a.tpot_slack:.2f}x; ttft slack {a.ttft_slack:.2f}x; quality gate {a.min_quality}")
    if any(r.get("cpu_tpot_ms") for r in rows_out):
        print("co-run CPU job (mean over its requests): " + ", ".join(
            f"{r['config']} {r['cpu_tpot_ms']:.1f} ms/tok" + (f" ({r['vs_naive_cpu_tpot']:.2f}x)" if r.get("vs_naive_cpu_tpot") else "")
            for r in rows_out if r.get("cpu_tpot_ms")))
    if any(r.get("gpu_tpot_ms") for r in rows_out):
        print("ck: GPU job tpot / heat from the first generated token on (vs baseline): " + ", ".join(
            f"{r['config']} {r['gpu_tpot_ms']:.1f} ms/tok" + (f" ({r['vs_naive_gpu_tpot']:.2f}x)" if r.get("vs_naive_gpu_tpot") else "")
            + (f", heat_post {r['heat_post_per_tok']:.2f}" + (f" ({r['vs_naive_heat_post']:.2f}x)" if r.get("vs_naive_heat_post") else "") if r.get("heat_post_per_tok") else "")
            for r in rows_out if r.get("gpu_tpot_ms")))
    if any(r.get("cpu_Bps_on") for r in rows_out):
        for r in rows_out:
            for k in ("cpu_Bps_on", "cpu_Bps_off"):
                r["vs_naive_" + k] = (r[k] / nv[k]) if nv and r.get(k) and nv.get(k) else None
        print("ck bursts: CPU job output rate during GPU bursts / between them (bytes/s; vs baseline, higher = faster): " + ", ".join(
            f"{r['config']} {r.get('cpu_Bps_on') or 0:.1f} / {r.get('cpu_Bps_off') or 0:.1f}"
            + (f" ({r['vs_naive_cpu_Bps_on']:.2f}x / {r['vs_naive_cpu_Bps_off']:.2f}x)" if r.get("vs_naive_cpu_Bps_on") and r.get("vs_naive_cpu_Bps_off") else "")
            for r in rows_out if r.get("cpu_Bps_on")))
    if any(not r["tail_complete"] for r in rows_out):
        print("WARNING: some tails were truncated (board did not return to baseline within --tail-max): their heat area is underestimated")
    if any(not r["window_ok"] for r in rows_out):
        print("WARNING: decode was shorter than --window + 20 s for some configs: dT_ss uses half of the decode instead")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        cols = ["config", "n", "kv", "mhz", "gpu_mhz_mean", "tpot_ms", "vs_naive_tpot", "prefill_s", "vs_naive_ttft", "total_s", "vs_naive_total", "switch_error_s", "dT_ss", "heat_area_per_tok", "heat_area_min", "heat_area_max",
                "vs_naive_heat", "peak_c", "max_cooling_state", "quality", "tail_complete", "window_ok", "cpu_tpot_ms", "vs_naive_cpu_tpot", "gpu_tpot_ms", "vs_naive_gpu_tpot", "heat_post_per_tok", "vs_naive_heat_post",
                "cpu_Bps_on", "vs_naive_cpu_Bps_on", "cpu_Bps_off", "vs_naive_cpu_Bps_off", "verdict"] \
            + UTIL_KEYS
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: S.fmt(r.get(k)) for k in cols})
    print("report written to", out_dir / "summary.csv")


def main():
    a = build_parser().parse_args()
    if a.analyze:
        d = Path(a.analyze)
        report([json.loads(l) for l in open(d / "runs.jsonl")], a, d)
        return
    configs = a.configs.split(",")
    for c in configs:
        parse_config(c)
    per = 3 + a.ntok * 0.09 / 60 + 5
    print(f"plan: configs {configs} x {a.repeat} round(s); prompt {a.prompt_chars} chars + {a.ntok} generated tokens, ctx {a.ctx}, A720 {a.cpus} x{a.threads}; "
          f"about {per:.0f}+ min per run (prefill + decode + cool-down tail) -> {len(configs) * a.repeat * per / 60:.1f}+ h")
    if a.dry_run:
        return
    if platform.machine() != "aarch64":
        sys.exit("run this on the board")
    for p in (a.model, a.llama_bin, a.run_stage1, a.prompt_src) + ((a.adapter,) if a.adapter and any(parse_config(c)[0] == "swap" for c in configs) else ()):
        if not Path(p).exists():
            sys.exit(f"missing: {p}")
    if subprocess.run(["sudo", "-n", "true"]).returncode:
        sys.exit("passwordless `sudo -n true` is needed (run_stage1.sh drops the page cache with it)")
    runs, done = [], set()
    if a.resume:
        out_dir = Path(a.resume)
        old = json.loads((out_dir / "meta.json").read_text())
        for k in ("ctx", "ntok", "prompt_chars", "configs", "repeat", "m", "g", "w", "sink", "reuse", "quality_swap"):
            if old["args"].get(k) != vars(a).get(k):
                sys.exit(f"--resume: flag {k} differs from the original run ({old['args'].get(k)!r} vs {vars(a).get(k)!r})")
        runs = [json.loads(l) for l in open(out_dir / "runs.jsonl")]
        done = {(r["config"], r["rep"]) for r in runs if "A" in r}
        old.setdefault("resumed_at", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
        (out_dir / "meta.json").write_text(json.dumps(old, indent=1))
        print(f"resuming {out_dir}: {len(done)} finished run(s) kept, the rest will run: "
              f"{[(c, r) for r in range(1, a.repeat + 1) for c in configs if (c, r) not in done]}", flush=True)
        prompt = out_dir / "prompt.txt"
    else:
        out_dir = Path(a.out_dir) if a.out_dir else REPO / "runs" / time.strftime("thermal_%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = out_dir / "prompt.txt"
        prompt.write_text(Path(a.prompt_src).read_text(errors="ignore")[:a.prompt_chars])
        (out_dir / "meta.json").write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "kernel": platform.release(),
                                                       "temp_start_c": S.temp_c(), "zones": [z.name for z in ZONES]}, indent=1))
    a.perf_mode = U.perf_mode(a.perf)
    print("bus_GBps (perf bus_access):", U.perf_note(a.perf_mode), flush=True)
    fq = S.Freq()
    def _on_signal(sig, _frame):                   # log who stopped us, then unwind through the finally block (frequencies restored)
        print(f"received signal {signal.Signals(sig).name}: stopping", flush=True)
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)   # a dropped ssh session must not stop a long run
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    try:
        for rep in range(1, a.repeat + 1):
            order = configs[(rep - 1) % len(configs):] + configs[:(rep - 1) % len(configs)]      # rotate so no config always runs first
            for name in order:
                if (name, rep) in done or (a.only and name not in a.only.split(",")):
                    continue
                res = run_one(name, rep, a, fq, out_dir, prompt)
                runs.append(res)
                with open(out_dir / "runs.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
    except KeyboardInterrupt:
        print("interrupted; restoring frequencies. Partial results are in", out_dir)
    finally:
        fq.restore()
    report(runs, a, out_dir)
    print("results:", out_dir)


if __name__ == "__main__":
    main()
