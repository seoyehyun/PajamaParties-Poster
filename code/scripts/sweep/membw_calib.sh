#!/bin/bash
# membw_calib.sh -- calibrate CPU PMU counters against known streaming traffic on the A720 cores.
#
# Why: this board has no usable uncore PMU (no CMN PMU, no DDR-controller PMU), so memory traffic can only be inferred from
# core PMU events. membw_probe streams a known number of bytes (read / write / copy / triad); for every PMU event this script
# reports how many streamed bytes one event stands for ("B_per_event"). That shows which events (LLC-load-misses =
# ll_cache_miss_rd, l3d/l2d refills, bus_access, ...) can be turned into GB/s, and with which factor.
# Differential method: each config runs LO and HI iterations; counter deltas cancel initialisation and process start.
#
# Needs: gcc (OpenMP); the 6.8 perf binary (the 6.6 kernel has no perf wrapper); sudo (perf_event_paranoid=2).
#   SUDO_PASS=... bash membw_calib.sh        (or passwordless sudo)
# Frequencies: all four A720 policies (0, 6, 8, 10) are pinned to the SAME value (min=max, performance) and restored on exit.
# Env: OUT RAW FREQS(kHz) THREADS PATTERNS CPUS ONE_CPU LO HI MIB PERF BIN
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/membw_calib.csv}; RAW=${RAW:-${OUT%.csv}.raw}
PERF=${PERF:-/usr/lib/linux-tools/6.8.0-147-generic/perf}
CPUS=${CPUS:-0,1,6,7,8,9,10,11}; ONE_CPU=${ONE_CPU:-6}
POLICIES="0 6 8 10"
FREQS=${FREQS:-"2200000 1200000"}                 # the lowest A720 policy max is 2.2 GHz, so 2.2 GHz is the common ceiling
PATTERNS=${PATTERNS:-"read write copy triad"}
THREADS=${THREADS:-"1 8"}
LO=${LO:-3}; HI=${HI:-23}; MIB=${MIB:-512}
PMU=armv8_pmuv3_0                                  # cpu0-1,6-11 (A720); armv8_pmuv3_1 is the A520s
BIN=${BIN:-/tmp/membw_probe}
CPUFREQ=/sys/devices/system/cpu/cpufreq
SETA="cpu_cycles bus_access ll_cache_miss_rd ll_cache_rd l3d_cache_refill l2d_cache_refill l2d_cache_wb"   # 6 programmable + cycle counter: no multiplexing
SETB="cpu_cycles mem_access stall_backend_mem stall_backend l3d_cache bus_cycles inst_retired"

sx() { if [ -n "${SUDO_PASS:-}" ]; then echo "$SUDO_PASS" | sudo -S -p "" "$@"; else sudo -n "$@"; fi; }
w() { echo "$2" | sudo -n tee "$1" >/dev/null; }          # NOPASSWD rule: tee on /sys/devices/system/cpu/cpufreq/policy*/*

[ -x "$PERF" ] || { echo "perf not found: $PERF" >&2; exit 1; }
sx true 2>/dev/null || { echo "sudo needed: set SUDO_PASS or allow passwordless sudo" >&2; exit 1; }
gcc -O3 -ffast-math -fopenmp -o "$BIN" "$HERE/membw_probe.c" || exit 1

declare -A O_MIN O_MAX O_GOV
for p in $POLICIES; do d=$CPUFREQ/policy$p; O_MIN[$p]=$(cat $d/scaling_min_freq); O_MAX[$p]=$(cat $d/scaling_max_freq); O_GOV[$p]=$(cat $d/scaling_governor); done
restore() { for p in $POLICIES; do d=$CPUFREQ/policy$p; w $d/scaling_max_freq ${O_MAX[$p]}; w $d/scaling_min_freq ${O_MIN[$p]}; w $d/scaling_governor ${O_GOV[$p]}; done; }
trap restore EXIT
pin_all() { for p in $POLICIES; do d=$CPUFREQ/policy$p; w $d/scaling_governor performance; w $d/scaling_min_freq 800000; w $d/scaling_max_freq $1; w $d/scaling_min_freq $1; done; }
temp_max() { cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1; }
verify_freq() {   # busy-load every A720 core for 0.4 s, then read the CPPC feedback frequency of each policy
  for c in ${CPUS//,/ }; do taskset -c $c timeout 0.4 sh -c 'while :; do :; done' & done; sleep 0.25
  line=""; for p in $POLICIES; do line="$line policy$p=$(cat $CPUFREQ/policy$p/cpuinfo_cur_freq)"; done; wait 2>/dev/null
  echo "FREQCHK target=$1 kHz:$line | temp_max=$(temp_max) mC" >&2
}
cpus_for() { if [ "$1" -eq 1 ]; then echo $ONE_CPU; else echo $CPUS | cut -d, -f1-$1; fi; }
evlist() { local o=""; for e in $1; do o="$o,$PMU/$e/"; done; echo "${o#,}"; }

: > "$RAW"
for f in $FREQS; do
  pin_all $f; verify_freq $f
  for p in $PATTERNS; do for t in $THREADS; do
    cpus=$(cpus_for $t)
    for set in A B; do
      [ $set = A ] && evs=$(evlist "$SETA") || evs=$(evlist "$SETB")
      for it in $LO $HI; do
        tmp=$(mktemp)
        # perf's counts go to stderr, redirected by this shell: "perf -o file" fails for root on a file we created (fs.protected_regular)
        out=$(sx env OMP_PROC_BIND=true "$PERF" stat -x, -e "$evs" -- taskset -c "$cpus" "$BIN" "$p" "$t" "$it" "$MIB" 2>"$tmp" | tail -1)
        IFS=, read -r _ _ _ loop bpi <<<"$out"
        # RAW: pattern,threads,freq_mhz,set,iters,loop_s,bytes_per_iter,event,count,pct_of_time_counted
        awk -F, -v pre="RAW,$p,$t,$((f / 1000)),$set,$it,$loop,$bpi" '$1 ~ /^[0-9]+$/ {n=split($3,a,"/"); print pre "," a[2] "," $1 "," $5}' "$tmp" >> "$RAW"
        rm -f "$tmp"
      done
    done
    echo "done $((f / 1000)) MHz $p x$t" >&2
  done; done
done

LO=$LO HI=$HI python3 - "$RAW" "$OUT" <<'PY'
import csv, os, sys, collections
raw, out = sys.argv[1], sys.argv[2]; LO, HI = int(os.environ["LO"]), int(os.environ["HI"])
d = collections.defaultdict(dict)       # (pattern,threads,freq) -> (set,iters) -> {"loop":..,"bpi":..,"ev":{name:count},"pct":min}
for l in open(raw):
    _, p, t, f, s, it, loop, bpi, ev, cnt, pct = l.strip().split(",")
    r = d[(p, int(t), int(f))].setdefault((s, int(it)), {"loop": float(loop), "bpi": float(bpi), "ev": {}, "pct": 100.0})
    r["ev"][ev] = int(cnt); r["pct"] = min(r["pct"], float(pct))
EV = ["bus_access", "ll_cache_miss_rd", "ll_cache_rd", "l3d_cache_refill", "l2d_cache_refill", "l2d_cache_wb", "mem_access", "l3d_cache", "bus_cycles", "stall_backend_mem", "stall_backend", "inst_retired", "cpu_cycles"]
rows = []
for (p, t, f), runs in sorted(d.items()):
    row = {"pattern": p, "threads": t, "freq_mhz": f}; mux = 100.0; tl = []
    for s in "AB":
        lo, hi = runs.get((s, LO)), runs.get((s, HI))
        if not lo or not hi: continue
        dt = hi["loop"] - lo["loop"]; streamed = (HI - LO) * hi["bpi"]; tl.append(dt); mux = min(mux, lo["pct"], hi["pct"])
        for ev, c in hi["ev"].items():
            dc = c - lo["ev"].get(ev, 0)
            if s == "B" and ev == "cpu_cycles": row["cycles_setB"] = dc; continue
            row[f"{ev}_Mps"] = round(dc / dt / 1e6, 1)
            row[f"B_per_{ev}"] = round(streamed / dc, 2) if dc > 0 else ""
        row["prog_GBps_" + s] = round(streamed / dt / 1e9, 2)
    if "prog_GBps_A" in row and "prog_GBps_B" in row: row["prog_GBps"] = round((row["prog_GBps_A"] + row["prog_GBps_B"]) / 2, 2)
    if row.get("cycles_setB") and "inst_retired_Mps" in row:
        dtb = (runs[("B", HI)]["loop"] - runs[("B", LO)]["loop"])
        row["ipc"] = round(row["inst_retired_Mps"] * 1e6 * dtb / row["cycles_setB"], 2)
        row["stall_backend_mem_frac"] = round(row["stall_backend_mem_Mps"] * 1e6 * dtb / row["cycles_setB"], 2)
    row["min_pct_counted"] = mux; rows.append(row)
cols = ["pattern", "threads", "freq_mhz", "prog_GBps"] + [f"B_per_{e}" for e in EV if e != "cpu_cycles"] + [f"{e}_Mps" for e in EV] + ["ipc", "stall_backend_mem_frac", "min_pct_counted"]
with open(out, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)
show = ["bus_access", "ll_cache_miss_rd", "l3d_cache_refill", "l2d_cache_refill", "l2d_cache_wb"]
print("bytes of streamed traffic represented by ONE event (B_per_event); 64 = one cache line per event")
print(f"{'pattern':7s} {'thr':>3s} {'MHz':>5s} {'GB/s':>6s} | " + " ".join(f"{e[:14]:>14s}" for e in show) + " | stallmem  ipc  mux%")
for r in rows:
    print(f"{r['pattern']:7s} {r['threads']:>3d} {r['freq_mhz']:>5d} {r.get('prog_GBps', 0):>6.1f} | " + " ".join(f"{r.get('B_per_' + e, ''):>14}" for e in show) + f" | {r.get('stall_backend_mem_frac', ''):>8} {r.get('ipc', ''):>4} {r['min_pct_counted']:>5.0f}")
print("wrote", out)
PY
