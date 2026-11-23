#!/usr/bin/env python3
"""util_sampler.py -- resource utilisation over a time window, shared by sweep_npu.py and joint/thermal_ab.py.

A background thread samples /proc/stat, /proc/diskstats and the Mali runtime-PM active time once per --period; optionally a system-wide
`perf stat -a -I 1000` counts bus_access on both CPU PMUs. window(lo, hi, suffix) turns the samples nearest [lo, hi] into:

  cpu_util_<group>    busy share of the group's cores (/proc/stat: everything but idle and iowait)
  cpu_iowait_<group>  iowait share of the group's cores (time a core sat idle with I/O outstanding: the KV-swap reads)
  disk_util_<label>   io_ticks share of that disk (iostat's %util; for an NVMe with many queues it saturates before the device does)
  disk_rd_MBps / disk_wr_MBps   summed over the whole disks (nvme*n*, sd?, mmcblk*)
  gpu_busy            Mali runtime-PM active share (the GPU suspends when idle; a coarse busy share, not shader occupancy)
  read_GBps_a720      l2d_cache_refill x 64 B on the A720 PMU: the PRIMARY DRAM read proxy (the 11/13 sweeps' pmu_read_GBps, exact in
                      membw_calib, within ~10% of model size x t/s in llama decode; the ~39-40 GB/s decode ceiling is in this unit)
  bus_GBps_a720 / bus_GBps_a520   bus_access x bus_bytes per PMU: an UPPER BOUND (~1.7x read_GBps in llama decode: it also counts
                      traffic that does not reach DRAM). The A520 PMU has no l2d_cache_refill, so the A520 side only has this one.
                      All CPU-side traffic ONLY: no uncore (DSU/CMN/DMC) PMU on this board, so NPU and GPU DRAM traffic is invisible.
                      perf needs kernel.perf_event_paranoid <= 0 or SUDO_PASS (perf is not in the password-less sudoers list); else omitted.
"""
import os
import subprocess
import threading
import time
from pathlib import Path

MALI_ACTIVE = Path("/sys/class/misc/mali0/device/power/runtime_active_time")
PMU_A720, PMU_A520 = "armv8_pmuv3_0", "armv8_pmuv3_1"       # cpu0-1,6-11 and cpu2-5
DEFAULT_PERF = "/usr/lib/linux-tools/6.8.0-147-generic/perf"


def cpulist(s):
    """'0-1,6-11' or '0,1,6' or 3 -> [0, 1, 6, ...]"""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def procstat():
    """{cpu: (busy, iowait, total)} jiffies from /proc/stat."""
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:]]
            iow = v[4] if len(v) > 4 else 0
            tot = sum(v[:8])                                  # user nice system idle iowait irq softirq steal (guest is inside user)
            out[int(f[0][3:])] = (tot - v[3] - iow, iow, tot)
    return out


def diskstats():
    """{whole disk: (io_ticks ms, sectors read, sectors written)}."""
    out = {}
    for line in open("/proc/diskstats"):
        f = line.split()
        n = f[2]
        if (n.startswith("nvme") and "p" not in n[4:]) or (n.startswith("sd") and len(n) == 3) or (n.startswith("mmcblk") and "p" not in n[6:]):
            out[n] = (int(f[12]), int(f[5]), int(f[9]))
    return out


def _active_ms(p):
    try:
        return int(Path(p).read_text())
    except (OSError, ValueError):
        return None


def perf_mode(perf_bin=DEFAULT_PERF):
    """'direct' (perf_event_paranoid <= 0), 'sudo' (SUDO_PASS set) or None (no bus_GBps)."""
    if not perf_bin or not Path(perf_bin).exists():
        return None
    try:
        if int(Path("/proc/sys/kernel/perf_event_paranoid").read_text()) <= 0:
            return "direct"
    except (OSError, ValueError):
        pass
    return "sudo" if os.environ.get("SUDO_PASS") else None


def perf_note(mode):
    return {"direct": "on (perf_event_paranoid <= 0)", "sudo": "on (SUDO_PASS)"}.get(
        mode, "OFF: needs `sudo sysctl kernel.perf_event_paranoid=-1` (until reboot) or SUDO_PASS")


class UtilLog:
    """groups: {name: [cpu, ...] or None (= all cores)}; disks: {label: device name prefix, e.g. 'nvme' or 'nvme0n1'} (max over matches)."""

    def __init__(self, groups, disks, perf_bin=DEFAULT_PERF, mode=None, bus_bytes=31.5, tmp_dir=None, period=1.0):
        self.groups, self.disks, self.bus_bytes, self.period = groups, disks, bus_bytes, period
        self.rows, self.stop = [], threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.perf_bin, self.mode = perf_bin, mode
        self.perf = self.perf_t0 = None
        self.perf_iv = []
        self.perf_file = Path(tmp_dir) / f"perf_bus_{os.getpid()}_{id(self)}.tmp" if (mode and tmp_dir) else None

    def _loop(self):
        while not self.stop.is_set():
            self.rows.append((time.time(), procstat(), diskstats(), _active_ms(MALI_ACTIVE)))
            self.stop.wait(self.period)

    def start(self):
        if self.perf_file:
            cmd = [self.perf_bin, "stat", "-a", "-x,", "-I", "1000", "-o", str(self.perf_file),
                   "-e", f"{PMU_A720}/l2d_cache_refill/,{PMU_A720}/bus_access/,{PMU_A520}/bus_access/", "--", "cat"]
            if self.mode == "sudo":
                cmd = ["sudo", "-S", "-p", ""] + cmd
            # `cat` keeps perf alive until its stdin closes, so a root perf stops without a kill; sudo -S reads the first line as password
            self.perf = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            if self.mode == "sudo":
                self.perf.stdin.write(os.environ["SUDO_PASS"] + "\n")
                self.perf.stdin.flush()
            self.perf_t0 = time.time()
        self.th.start()
        return self

    def end(self):
        self.stop.set()
        self.th.join(timeout=5)
        if self.perf:
            try:
                self.perf.stdin.close()
                self.perf.wait(timeout=10)
            except Exception:      # noqa: BLE001
                self.perf.kill()
            self.perf_iv = self._read_perf()

    def _read_perf(self):
        """[(t_end, "pmu/event", count)]; perf's time column is seconds since perf started."""
        if not self.perf_file or not self.perf_file.exists():
            return []
        out = []
        for line in self.perf_file.read_text().splitlines():
            p = line.split(",")
            if len(p) >= 4 and p[1].strip().isdigit():
                try:
                    out.append((self.perf_t0 + float(p[0]), "/".join(p[3].split("/")[:2]), int(p[1])))
                except ValueError:
                    pass
        self.perf_file.unlink(missing_ok=True)
        return out

    def window(self, lo, hi, suffix=""):
        """Utilisation over [lo, hi] (call after end(); may be called for several windows)."""
        res = {}
        rs = [r for r in self.rows if lo - self.period / 2 <= r[0] <= hi + self.period / 2]
        if len(rs) >= 2:
            (t0, c0, d0, m0), (t1, c1, d1, m1) = rs[0], rs[-1]
            dt = t1 - t0
            for name, cpus in self.groups.items():
                cs = [i for i in (cpus if cpus is not None else c1) if i in c0 and i in c1]
                tot = sum(c1[i][2] - c0[i][2] for i in cs)
                if tot:
                    res[f"cpu_util_{name}"] = sum(c1[i][0] - c0[i][0] for i in cs) / tot
                    res[f"cpu_iowait_{name}"] = sum(c1[i][1] - c0[i][1] for i in cs) / tot
            for label, pre in self.disks.items():
                ticks = [d1[n][0] - d0[n][0] for n in d1 if n.startswith(pre) and n in d0]
                if ticks:
                    res[f"disk_util_{label}"] = max(ticks) / 1000.0 / dt
            res["disk_rd_MBps"] = sum(d1[n][1] - d0[n][1] for n in d1 if n in d0) * 512 / 1e6 / dt
            res["disk_wr_MBps"] = sum(d1[n][2] - d0[n][2] for n in d1 if n in d0) * 512 / 1e6 / dt
            if m0 is not None and m1 is not None:
                res["gpu_busy"] = (m1 - m0) / 1000.0 / dt
        iv = [x for x in self.perf_iv if lo + 1.0 <= x[0] <= hi]          # whole 1 s intervals inside the window
        if iv:
            secs = len({round(x[0], 1) for x in iv})
            for ev, key, nbytes in ((f"{PMU_A720}/l2d_cache_refill", "read_GBps_a720", 64.0), (f"{PMU_A720}/bus_access", "bus_GBps_a720", self.bus_bytes),
                                    (f"{PMU_A520}/bus_access", "bus_GBps_a520", self.bus_bytes)):
                if any(x[1] == ev for x in iv):
                    res[key] = sum(x[2] for x in iv if x[1] == ev) * nbytes / secs / 1e9
        return {k + suffix: v for k, v in res.items()}
