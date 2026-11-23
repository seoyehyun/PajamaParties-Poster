#!/usr/bin/env python3
"""Post-process one sweep_ri.sh run directory.
usage: ri_post.py <run_dir> <cpu_list> [<cpu_want_mhz> [<tol>]]

Reads  <dir>/freq.want  expected value of every sampled frequency node (one per line)
       <dir>/freq.log   "<epoch> <cur1> <cur2> ..." once per second during the run
       <dir>/ctr.log    "T <epoch>" then "c<N> ref:<n> del:<n>" per pinned core (CPPC feedback counters), once per second
       <dir>/util.log   "T <epoch>" followed by the /proc/stat cpu lines, once per second
       <dir>/err.txt    llama_perf output (prompt eval time / eval time), used to split prefill and decode
Prints one CSV fragment:  freq_ok,freq_real_mhz,util_sel_p,util_sel_d,util_all_d,iowait_d
  freq_ok       0 if any frequency node sample differs from its expected value, or, when <cpu_want_mhz> is given and the
                counters have data, if the real frequency of any pinned core is off by more than <tol> (default 0.02);
                1 if all checks pass, empty if there was nothing to check
  freq_real_mhz mean over the pinned cores of delta del / delta ref over the run (the counters only advance while the core
                runs, so this is the frequency while busy). scaling_cur_freq only echoes the request, see
                docs/sweep/results/freq_steps_20261110.csv
  util_sel_*    mean busy % over the pinned cores (cpu_list), prefill (p) / decode (d) window
  util_all_d    mean busy % over all cores in the decode window
  iowait_d      mean %iowait over all cores in the decode window
The windows are cut from the END of the sample log: decode = last <eval ms>, prefill = the <prompt ms> before it.
This is an approximation (model load is before both, teardown is negligible), good to about one sample period.
"""
import re
import sys
from pathlib import Path


def cpu_set(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def read_freq(d):
    want_f, log_f = d / "freq.want", d / "freq.log"
    if not want_f.exists() or not log_f.exists():
        return ""
    want = [int(x) for x in want_f.read_text().split()]
    n = 0
    for line in log_f.read_text().splitlines():
        f = line.split()
        if len(f) != len(want) + 1:
            continue
        n += 1
        if [int(x) for x in f[1:]] != want:
            return "0"
    return "1" if n else ""


def read_real_mhz(d):
    """-> {cpu: real MHz} from the first and last sample of ctr.log with delta ref > 0 (cores that never ran are left out)"""
    p = d / "ctr.log"
    if not p.exists():
        return {}
    first, last = {}, {}
    for line in p.read_text().splitlines():
        m = re.match(r"c(\d+)\s+ref:(\d+)\s+del:(\d+)", line)
        if not m:
            continue
        c, ref, dele = int(m[1]), int(m[2]), int(m[3])
        first.setdefault(c, (ref, dele))
        last[c] = (ref, dele)
    out = {}
    for c, (r0, d0) in first.items():
        r1, d1 = last[c]
        if r1 > r0:
            out[c] = (d1 - d0) / (r1 - r0) * 1000.0   # ref counts at ~1 GHz: del/ref in GHz -> MHz
    return out


def read_util(d):
    """-> list of (epoch, {cpu: (busy, idle, iowait, total)}, all=(...)) cumulative counters"""
    p = d / "util.log"
    if not p.exists():
        return []
    samples, cur = [], None
    for line in p.read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "T":
            cur = {"t": float(f[1]), "cpu": {}}
            samples.append(cur)
        elif cur is not None and f[0].startswith("cpu"):
            v = [int(x) for x in f[1:9]]
            user, nice, system, idle, iowait, irq, softirq, steal = v
            total = sum(v)
            key = "all" if f[0] == "cpu" else int(f[0][3:])
            cur["cpu"][key] = (total - idle - iowait, idle, iowait, total)
    return samples


def window_mean(samples, t0, t1, cpus, field):
    """mean busy% (field='busy') or iowait% over [t0, t1] for cpus ('all' = aggregate line)"""
    a = [s for s in samples if t0 <= s["t"] <= t1]
    if len(a) < 2:
        return None
    first, last = a[0], a[-1]
    vals = []
    for c in cpus:
        if c not in first["cpu"] or c not in last["cpu"]:
            continue
        f, l = first["cpu"][c], last["cpu"][c]
        dt = l[3] - f[3]
        if dt <= 0:
            continue
        num = (l[0] - f[0]) if field == "busy" else (l[2] - f[2])
        vals.append(100.0 * num / dt)
    return sum(vals) / len(vals) if vals else None


def fmt(x):
    return "" if x is None else f"{x:.1f}"


def main():
    d = Path(sys.argv[1])
    pinned = cpu_set(sys.argv[2])
    freq_ok = read_freq(d)
    real = read_real_mhz(d)
    real_mhz = sum(real.values()) / len(real) if real else None
    if len(sys.argv) > 3 and sys.argv[3] and real:
        want, tol = float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
        off = [c for c, v in real.items() if abs(v - want) / want > tol]
        if off:
            freq_ok = "0"
        elif freq_ok == "":
            freq_ok = "1"

    err = (d / "err.txt").read_text(errors="replace") if (d / "err.txt").exists() else ""
    m = re.search(r"prompt eval time\s*=\s*([\d.]+) ms", err)
    pp_ms = float(m[1]) if m else None
    m = re.search(r"common_perf_print:\s+eval time\s*=\s*([\d.]+) ms", err)
    tg_ms = float(m[1]) if m else None

    samples = read_util(d)
    up = ud = ua = iw = None
    if samples and pp_ms and tg_ms:
        t_end = samples[-1]["t"]
        t_dec = t_end - tg_ms / 1000.0
        t_pre = t_dec - pp_ms / 1000.0
        up = window_mean(samples, t_pre, t_dec, pinned, "busy")
        ud = window_mean(samples, t_dec, t_end, pinned, "busy")
        ua = window_mean(samples, t_dec, t_end, ["all"], "busy")
        iw = window_mean(samples, t_dec, t_end, ["all"], "iowait")
    print(",".join([freq_ok, "" if real_mhz is None else f"{real_mhz:.0f}", fmt(up), fmt(ud), fmt(ua), fmt(iw)]))


if __name__ == "__main__":
    main()
