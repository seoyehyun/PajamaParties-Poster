#!/usr/bin/env bash
# Step probe for CPU frequency pinning (docs/sweep/measurement-control-plan.md 10-2). Run on the board.
# Pins each cpufreq policy to every step from 800 MHz to its max, loads one core of the policy and
# records the requested value, scaling_cur_freq, cpuinfo_cur_freq and the real average frequency
# from the CPPC feedback counters (delta del / delta ref, GHz). Original min/max/governor are restored on exit.
# usage: scripts/sweep/freq_steps.sh
# env:   POLICIES  default "0 2 6 8 10"      STEP_KHZ default 100000
#        DUR       load seconds per step (default 3)   SETTLE seconds after pinning (default 1)
#        COOL_MC   wait below this temperature (milli-C) before each step, default 55000
#        OUT       csv path, default freq_steps_<date>.csv
# Needs the sudoers rule for tee on cpufreq/policy*/* (docs/sweep/board-freq-config.md 8-1). Do not run while another sweep runs.
set -uo pipefail
POLICIES=${POLICIES:-"0 2 6 8 10"}; STEP=${STEP_KHZ:-100000}; DUR=${DUR:-3}; SETTLE=${SETTLE:-1}
COOL_MC=${COOL_MC:-55000}
OUT=${OUT:-freq_steps_$(date +%Y%m%d_%H%M).csv}
CF=/sys/devices/system/cpu/cpufreq
declare -A MIN MAX GOV

for p in $POLICIES; do
  d=$CF/policy$p
  [[ -d $d ]] || { echo "no policy$p" >&2; exit 1; }
  MIN[$p]=$(cat "$d/scaling_min_freq"); MAX[$p]=$(cat "$d/scaling_max_freq"); GOV[$p]=$(cat "$d/scaling_governor")
done
sudo -n true || { echo "passwordless sudo is required" >&2; exit 1; }

LOADPID=
restore() {
  [[ -n $LOADPID ]] && kill "$LOADPID" 2>/dev/null
  for p in $POLICIES; do
    d=$CF/policy$p
    echo "${MAX[$p]}" | sudo tee "$d/scaling_max_freq" >/dev/null 2>&1
    echo "${MIN[$p]}" | sudo tee "$d/scaling_min_freq" >/dev/null 2>&1
    echo "${GOV[$p]}" | sudo tee "$d/scaling_governor" >/dev/null 2>&1
  done
  echo "restored: $(for p in $POLICIES; do echo -n "policy$p=$(cat $CF/policy$p/scaling_governor) "; done)" >&2
}
trap restore EXIT INT TERM

maxtemp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
cooling() { cat /sys/class/thermal/cooling_device*/cur_state 2>/dev/null | awk '{s+=$1}END{print s+0}'; }
ctr() { sed -E 's/ref:([0-9]+) del:([0-9]+)/\1 \2/' "$1"; }

echo "policy,core,req_khz,scaling_cur_khz,cpuinfo_cur_khz,real_ghz,delta_ref,temp_start_mC,temp_end_mC,cooling_state,pinned_ok" > "$OUT"

for p in $POLICIES; do
  d=$CF/policy$p
  core=$(awk '{print $1}' "$d/affected_cpus")
  f=800000; steps=()
  while (( f < MAX[$p] )); do steps+=("$f"); f=$((f + STEP)); done
  steps+=("${MAX[$p]}")
  for f in "${steps[@]}"; do
    while t=$(maxtemp); [[ ${t:-0} -gt $COOL_MC ]]; do echo "  cooling: $t" >&2; sleep 10; done
    # reset the range first so min <= max holds in every order, then pin
    echo "${MAX[$p]}" | sudo tee "$d/scaling_max_freq" >/dev/null
    echo "${MIN[$p]}" | sudo tee "$d/scaling_min_freq" >/dev/null
    echo userspace | sudo tee "$d/scaling_governor" >/dev/null
    echo "$f" | sudo tee "$d/scaling_min_freq" >/dev/null
    echo "$f" | sudo tee "$d/scaling_max_freq" >/dev/null
    echo "$f" | sudo tee "$d/scaling_setspeed" >/dev/null
    t0=$(maxtemp)
    taskset -c "$core" sh -c 'while :; do :; done' & LOADPID=$!
    sleep "$SETTLE"
    read -r r1 d1 <<< "$(ctr /sys/devices/system/cpu/cpu$core/acpi_cppc/feedback_ctrs)"
    sleep "$DUR"
    read -r r2 d2 <<< "$(ctr /sys/devices/system/cpu/cpu$core/acpi_cppc/feedback_ctrs)"
    sc=$(cat "$d/scaling_cur_freq"); ci=$(cat "/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_cur_freq" 2>/dev/null || echo NA)
    kill "$LOADPID" 2>/dev/null; wait "$LOADPID" 2>/dev/null; LOADPID=
    dr=$((r2 - r1))
    real=$(awk -v a="$d2" -v b="$d1" -v c="$dr" 'BEGIN{ if (c>0) printf "%.4f", (a-b)/c; else print "NA" }')
    ok=$([[ $sc -eq $f ]] && echo 1 || echo 0)
    echo "$p,$core,$f,$sc,$ci,$real,$dr,$t0,$(maxtemp),$(cooling),$ok" | tee -a "$OUT"
  done
done
echo "done: $OUT"
