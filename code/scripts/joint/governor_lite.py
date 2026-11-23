#!/usr/bin/env python3
"""governor_lite.py -- dominant-resource frequency management at run time: sense the state, look the setting up in the
pre-computed table (docs/joint/precomputed-table.md + docs/report-runs/results-20261121.md), apply it, log every decision.

State, sensed every --period seconds (0.5 s by default):
  cpu   idle | prefill | decode   A720 busy share (/proc/stat over --cpus) > --busy, and the CPU job's output file (--watch-out):
                                  empty = prefill, non-empty = decode (llama-completion writes tokens as they are produced; the same
                                  first-token signal thermal_ab.py uses for its phase_ configs)
  gpu   on | off                  Mali runtime-PM active share > --busy
  npu   on | off                  NPU devfreq runtime-PM active share > --busy (logged; no NPU rule: NPU heat is below the sensor resolution)
  kv    ram | swap                given (--kv): KV placement is a launch option of the CPU job and cannot change while it runs
A new setting is applied only after the same state has been seen --stable times in a row (no flapping on a single sample).

Table (rule -> A720, GPU; None = the default governor):
  cpu prefill (any)            A720 def, GPU def    prefill is compute bound: lowering the A720 costs TTFT 1:1 (+50.6 % at 1.5 GHz)
  cpu decode, gpu off, ram     A720 1500, GPU def   decode-only 1.5 GHz: heat 0.87, TTFT 1.00, TPOT 1.05 (11/14, E6 0.88)
  cpu decode, gpu off, swap    A720 1500, GPU def   11/21 rerun (n=2): heat after the first token 0.78-0.84 of KV swap at default, TPOT 1.02 of naive
                                                    (was 'default' until 11/21 from one 11/14 run, 0.904 vs 0.903; the 11/21 gov run used that rule)
  cpu decode, gpu on,  swap    A720 1500, GPU def   E7: heat 0.81 (0.55 after the first token), CPU job 0.97, GPU job 0.79, TTFT 1.05
  cpu decode, gpu on,  ram     A720 1500, GPU def   nearest measured cell (A720 1.5 GHz for the whole request: heat 0.71, CPU job 1.07)
  (the GPU is left on its own governor: pinning it to 1000 MHz in co-run changed nothing, heat 1.01; E7 had it pinned)
  cpu idle,   gpu on           A720 def,  GPU def   GPU alone: lowering the GPU raised the heat (350-800 MHz: 1.09-1.44); leave it
  cpu idle,   gpu off          A720 def,  GPU def

Use it from thermal_ab.py (configs ck_<ram|swap>_gov) or alone, e.g. for a live demo in a terminal:
  python3 scripts/joint/governor_lite.py --kv swap --watch-out runs/<run>/out.txt --log /tmp/gov.csv
It needs the NOPASSWD sudo rules of thermal_ab.py (tee on cpufreq/devfreq). Frequencies are restored on exit, Ctrl-C and SIGTERM.
Log: one CSV row per poll (gov.csv) + one JSON line per applied change (gov_events.jsonl); scripts/joint/gov_replay.py animates them.
"""
import argparse
import csv
import json
import signal
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sweep"))
import sweep_cpufreq as S   # noqa: E402
import util_sampler as U    # noqa: E402

NPU_ACTIVE = Path("/sys/class/devfreq/CIXH4000:00/device/power/runtime_active_time")

# (cpu phase, gpu on, kv) -> (A720 MHz or None, GPU MHz or None, rule name, evidence)
def decide(cpu, gpu, kv):
    if cpu == "prefill":
        return None, None, "prefill: keep max", "prefill is compute bound; A720 1.5 GHz costs TTFT +50.6 %"
    if cpu == "decode" and not gpu:
        if kv == "ram":
            return 1500, None, "CPU decode: A720 1.5 GHz", "decode-only 1.5 GHz: heat 0.87, TTFT 1.00, TPOT 1.05"
        return 1500, None, "CPU decode, KV swap: A720 1.5 GHz", "11/21 n=2: heat after 1st token 0.78-0.84 of swap default, TPOT 1.02"
    if cpu == "decode" and gpu:
        if kv == "swap":
            return 1500, None, "co-run decode: A720 1.5 GHz", "E7: heat 0.81, CPU job 0.97, GPU job 0.79, TTFT 1.05"
        return 1500, None, "co-run decode: A720 1.5 GHz (nearest cell)", "whole-request 1.5 GHz: heat 0.71, CPU job 1.07"
    if gpu:
        return None, None, "GPU only: leave", "lowering the GPU raised heat (1.09-1.44)"
    return None, None, "idle", ""


class Sensors:
    """Reads the board. Every method is cheap (sysfs/procfs only)."""
    def __init__(self, cpus, watch_out):
        self.cpus = U.cpulist(cpus)
        self.watch_out = Path(watch_out) if watch_out else None
        self.prev = None

    @staticmethod
    def _ms(p):
        try:
            return int(S.rd(p))
        except Exception:      # noqa: BLE001 - node missing on this kernel
            return None

    def read(self):
        now = time.time()
        st = U.procstat()
        cur = {"t": now, "st": st, "gpu": self._ms(U.MALI_ACTIVE), "npu": self._ms(NPU_ACTIVE)}
        out = {"t": now, "a720_util": None, "gpu_busy": None, "npu_busy": None}
        if self.prev:
            p = self.prev
            busy = sum(st[c][0] - p["st"][c][0] for c in self.cpus if c in st and c in p["st"])
            tot = sum(st[c][2] - p["st"][c][2] for c in self.cpus if c in st and c in p["st"])
            out["a720_util"] = busy / tot if tot else 0.0
            dt_ms = (now - p["t"]) * 1000
            for k in ("gpu", "npu"):
                if cur[k] is not None and p[k] is not None and dt_ms > 0:
                    out[k + "_busy"] = min(1.0, max(0.0, (cur[k] - p[k]) / dt_ms))
        self.prev = cur
        out["first_token"] = bool(self.watch_out and self.watch_out.exists() and self.watch_out.stat().st_size > 0)
        try:
            out["cpu6_mhz"] = int(S.rd(S.CPUFREQ / "policy6/cpuinfo_cur_freq")) // 1000
            out["gpu_mhz"] = int(S.rd(S.GPU_DEV / "cur_freq")) // 1000000
        except Exception:      # noqa: BLE001
            out["cpu6_mhz"] = out["gpu_mhz"] = None
        out["temp_c"] = S.temp_c()
        return out


class Actuator:
    """Pins / releases the A720 policies and the GPU through sweep_cpufreq.Freq; only writes when the target changes."""
    def __init__(self, fq):
        self.fq, self.a720, self.gpu = fq, "unset", "unset"

    def set(self, a720, gpu):
        if a720 != self.a720:
            if a720 is None:
                for p, (mn, mx, gov) in self.fq.cpu.items():
                    d = S.CPUFREQ / f"policy{p}"
                    S.wr(d / "scaling_max_freq", mx); S.wr(d / "scaling_min_freq", mn); S.wr(d / "scaling_governor", gov)
            else:
                self.fq.pin_cpu(a720 * 1000)
            self.a720 = a720
        if gpu != self.gpu:
            if gpu is None:
                self.fq.unpin_gpu()
            else:
                self.fq.pin_gpu(gpu * 1000000)
            self.gpu = gpu


class Governor:
    def __init__(self, actuator, sensors, kv, log_path, period=0.5, stable=2, busy=0.3, t0=None):
        self.act, self.sen, self.kv, self.period, self.stable, self.busy = actuator, sensors, kv, period, stable, busy
        self.log_path = Path(log_path)
        self.ev_path = self.log_path.with_name(self.log_path.stem + "_events.jsonl")
        self.t0 = t0
        self.stop_ev, self.th = threading.Event(), None
        self.applied, self.cand, self.cand_n, self.cpu_seen = None, None, 0, False
        self.events = []

    def state(self, s):
        cpu_on = s["a720_util"] is not None and s["a720_util"] > self.busy
        self.cpu_seen = self.cpu_seen or cpu_on or s["first_token"]
        if s["first_token"] and cpu_on:
            cpu = "decode"
        elif cpu_on:
            cpu = "prefill"
        else:
            cpu = "idle"
        gpu = bool(s["gpu_busy"] is not None and s["gpu_busy"] > self.busy)
        return cpu, gpu

    def step(self):
        s = self.sen.read()
        self.t0 = self.t0 or s["t"]
        if s["a720_util"] is None:                     # first sample: no deltas yet
            return s
        cpu, gpu = self.state(s)
        target = decide(cpu, gpu, self.kv)
        key = (cpu, gpu)
        if key == self.cand:
            self.cand_n += 1
        else:
            self.cand, self.cand_n = key, 1
        if self.cand_n >= self.stable and (self.applied is None or self.applied[0] != key):
            self.act.set(target[0], target[1])
            ev = {"t": s["t"], "t_rel": round(s["t"] - self.t0, 2), "cpu": cpu, "gpu": gpu, "kv": self.kv,
                  "a720_mhz": target[0], "gpu_mhz": target[1], "rule": target[2], "evidence": target[3]}
            self.applied = (key, target)
            self.events.append(ev)
            with open(self.ev_path, "a") as f:
                f.write(json.dumps(ev) + "\n")
            print(f"[gov {ev['t_rel']:7.1f}s] {cpu}{' + GPU' if gpu else ''} -> {target[2]}  "
                  f"(A720 {target[0] or 'def'}, GPU {target[1] or 'def'})  {target[3]}", flush=True)
        a = self.applied[1] if self.applied else (None, None, "", "")
        row = {"t": round(s["t"], 2), "t_rel": round(s["t"] - self.t0, 2), "a720_util": round(s["a720_util"], 3),
               "gpu_busy": None if s["gpu_busy"] is None else round(s["gpu_busy"], 3),
               "npu_busy": None if s["npu_busy"] is None else round(s["npu_busy"], 3), "first_token": int(s["first_token"]),
               "cpu_phase": cpu, "gpu_on": int(gpu), "kv": self.kv, "a720_set_mhz": a[0], "gpu_set_mhz": a[1], "rule": a[2],
               "cpu6_mhz": s["cpu6_mhz"], "gpu_mhz": s["gpu_mhz"], "temp_c": s["temp_c"]}
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
        return s

    def _loop(self):
        while not self.stop_ev.is_set():
            t = time.time()
            try:
                self.step()
            except Exception as e:     # noqa: BLE001 - a sensor glitch must not kill the run; log and go on
                print("governor step failed:", e, file=sys.stderr, flush=True)
            self.stop_ev.wait(max(0.0, self.period - (time.time() - t)))

    def start(self):
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def stop(self):
        self.stop_ev.set()
        if self.th:
            self.th.join(timeout=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kv", choices=["ram", "swap"], required=True, help="KV placement of the CPU job (fixed at its launch)")
    ap.add_argument("--watch-out", help="the CPU job's token output file (prefill/decode switch); without it a busy A720 counts as prefill")
    ap.add_argument("--cpus", default="0-1,6-11", help="A720 cores of the CPU job")
    ap.add_argument("--log", default="gov.csv")
    ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--stable", type=int, default=2, help="polls a new state must persist before it is applied")
    ap.add_argument("--busy", type=float, default=0.3, help="busy share above which a resource counts as active")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until Ctrl-C)")
    a = ap.parse_args()
    fq = S.Freq()
    gov = Governor(Actuator(fq), Sensors(a.cpus, a.watch_out), a.kv, a.log, a.period, a.stable, a.busy)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        gov.start()
        t0 = time.time()
        while not a.duration or time.time() - t0 < a.duration:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        gov.stop()
        fq.restore()
        print("frequencies restored; log:", a.log)


if __name__ == "__main__":
    main()
