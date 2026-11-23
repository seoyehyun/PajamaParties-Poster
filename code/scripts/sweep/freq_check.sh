#!/usr/bin/env bash
# Frequency-control smoke test for Orange Pi 6 Plus. Needs root:  sudo scripts/sweep/freq_check.sh
# For each CPU policy: switch to userspace, pin a low then a high step, verify cur freq, restore.
# Then the same for NPU (CIXH4000:00) and GPU (CIXH5000:00) devfreq. Everything is restored on exit.
# usage: sudo scripts/sweep/freq_check.sh [log_file]     (default: freq_check.log in the current directory)
set -uo pipefail
[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0" >&2; exit 1; }
LOG=${1:-freq_check.log}; : > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }

declare -A SAVE
restore() {
  for p in /sys/devices/system/cpu/cpufreq/policy*; do
    [[ -n ${SAVE[$p.gov]:-} ]] || continue
    echo "${SAVE[$p.min]}" > $p/scaling_min_freq 2>/dev/null
    echo "${SAVE[$p.max]}" > $p/scaling_max_freq 2>/dev/null
    echo "${SAVE[$p.gov]}" > $p/scaling_governor 2>/dev/null
  done
  for d in /sys/class/devfreq/*; do
    [[ -n ${SAVE[$d.gov]:-} ]] || continue
    echo "${SAVE[$d.gov]}" > $d/governor 2>/dev/null
    echo "${SAVE[$d.min]}" > $d/min_freq 2>/dev/null
    echo "${SAVE[$d.max]}" > $d/max_freq 2>/dev/null
  done
  say "restored original governors and limits"
}
trap restore EXIT

say "== $(date -Is)  kernel $(uname -r)"

# ---- CPU ----
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  n=$(basename $p); cpus=$(cat $p/related_cpus)
  SAVE[$p.gov]=$(cat $p/scaling_governor); SAVE[$p.min]=$(cat $p/scaling_min_freq); SAVE[$p.max]=$(cat $p/scaling_max_freq)
  hw_min=$(cat $p/cpuinfo_min_freq); hw_max=$(cat $p/cpuinfo_max_freq)
  say "-- $n cpus=[$cpus] hw=$hw_min..$hw_max  was gov=${SAVE[$p.gov]} min=${SAVE[$p.min]} max=${SAVE[$p.max]}"
  if ! echo userspace > $p/scaling_governor 2>>"$LOG"; then say "   FAIL: cannot set userspace governor"; continue; fi
  for f in $hw_min 1200000 1800000 $hw_max; do
    (( f > hw_max )) && continue
    # widen limits first, then set (order matters when going down vs up)
    echo $hw_min > $p/scaling_min_freq 2>>"$LOG"; echo $hw_max > $p/scaling_max_freq 2>>"$LOG"
    echo $f > $p/scaling_setspeed 2>>"$LOG"
    sleep 0.5
    cur=$(cat $p/scaling_cur_freq); hwcur=$(cat $p/cpuinfo_cur_freq 2>/dev/null || echo n/a)
    ok=$([[ $cur == $f ]] && echo OK || echo MISMATCH)
    say "   set $f -> scaling_cur=$cur cpuinfo_cur=$hwcur  $ok"
  done
  # fallback test: does min=max pinning work when setspeed does not?
  echo performance > $p/scaling_governor 2>>"$LOG"
  echo 1200000 > $p/scaling_max_freq 2>>"$LOG"; echo 1200000 > $p/scaling_min_freq 2>>"$LOG"
  sleep 0.5; say "   min=max=1200000 (performance) -> scaling_cur=$(cat $p/scaling_cur_freq)"
done

# ---- devfreq ----
for d in /sys/class/devfreq/*; do
  n=$(basename $d)
  SAVE[$d.gov]=$(cat $d/governor); SAVE[$d.min]=$(cat $d/min_freq); SAVE[$d.max]=$(cat $d/max_freq)
  freqs=$(cat $d/available_frequencies); lo=${freqs%% *}; hi=${freqs##* }
  say "-- $n driver=$(basename "$(readlink $d/device/driver)") freqs=[$freqs] was gov=${SAVE[$d.gov]}"
  for f in $lo $hi; do
    echo performance > $d/governor 2>>"$LOG"
    # min=max pinning (raise max first when going up, lower min first when going down)
    if (( f >= $(cat $d/max_freq) )); then echo $f > $d/max_freq 2>>"$LOG"; echo $f > $d/min_freq 2>>"$LOG"
    else echo $f > $d/min_freq 2>>"$LOG"; echo $f > $d/max_freq 2>>"$LOG"; fi
    sleep 0.5
    cur=$(cat $d/cur_freq); say "   pin $f -> cur_freq=$cur  $([[ $cur == $f ]] && echo OK || echo MISMATCH)"
    echo $lo > $d/min_freq 2>/dev/null; echo $hi > $d/max_freq 2>/dev/null
  done
done

# ---- energy model / OPP (needs root for debugfs) ----
say "-- debugfs"
ls /sys/kernel/debug 2>&1 | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"
ls /sys/kernel/debug/energy_model 2>&1 | head -5 | tee -a "$LOG"
ls /sys/kernel/debug/opp 2>&1 | head -5 | tee -a "$LOG"
ls /sys/kernel/debug/aipu /sys/kernel/debug/mali* 2>&1 | head -5 | tee -a "$LOG"
say "== done, log: $LOG"
