#!/usr/bin/env bash
# Run any command at each NPU frequency and log wall time, NPU active time and temperature. Run on the board. NOT yet run.
# usage: scripts/npu/llm/npu_freq_run.sh "<MHz list>" REPS -- <command ...>
#   e.g. npu_freq_run.sh "400 600 800 1200" 3 -- python3 qwen38_generate.py "Hello" --max-new 32
# Output: one CSV line per run on stdout (header first), the command's own output goes to <OUT>/<freq>_r<rep>.log
# env:   OUT        log dir, default npu_freq_run_<date>
#        COOL_MC    wait until the hottest thermal zone is below this (milli-C), default 55000
#        CPUS       taskset list for the command, default none (route B pins itself with QWEN_CPUS)
# The command prints its own tokens/s; grep it out of the logs afterwards. Frequencies are the NPU devfreq OPPs
# (400 600 800 1200 MHz, docs/npu/npu-setup.md). Order is shuffled, the original governor and set_freq are restored on exit.
# Needs the sudoers rule for tee on /sys/class/devfreq/*/* (docs/sweep/board-freq-config.md 8-1).
set -uo pipefail
freqs=$1; reps=$2; shift 2
[[ ${1:-} == -- ]] && shift
[[ $# -gt 0 ]] || { echo "usage: $0 \"<MHz list>\" REPS -- <command ...>" >&2; exit 1; }
D=/sys/class/devfreq/CIXH4000:00
OUT=${OUT:-npu_freq_run_$(date +%Y%m%d_%H%M)}; COOL_MC=${COOL_MC:-55000}; mkdir -p "$OUT"
gov0=$(cat $D/governor); set0=$(cat $D/userspace/set_freq 2>/dev/null || true); cur0=$(cat $D/cur_freq)

restore() {
  echo "$gov0" | sudo -n tee $D/governor >/dev/null 2>&1 || true
  case "$set0" in ''|*[!0-9]*) v=$cur0 ;; *) v=$set0 ;; esac
  [[ -e $D/userspace/set_freq ]] && echo "$v" | sudo -n tee $D/userspace/set_freq >/dev/null 2>&1 || true
  echo "restored governor=$(cat $D/governor) cur=$(cat $D/cur_freq)" >&2
}
trap restore EXIT INT TERM
sudo -n true || { echo "passwordless sudo needed" >&2; exit 1; }
echo userspace | sudo -n tee $D/governor >/dev/null || { echo "cannot switch NPU governor" >&2; exit 1; }

maxtemp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
echo "freq_mhz,rep,exit,wall_s,npu_active_ms,cur_freq_hz_start,cur_freq_hz_end,temp_start_mC,temp_end_mC,log"
sched=(); for f in $freqs; do for r in $(seq 1 "$reps"); do sched+=("$f:$r"); done; done
mapfile -t sched < <(printf '%s\n' "${sched[@]}" | shuf)
for item in "${sched[@]}"; do
  f=${item%%:*}; r=${item##*:}
  while t=$(maxtemp); [[ ${t:-0} -gt $COOL_MC ]]; do echo "  cooling: $t" >&2; sleep 10; done
  echo $((f * 1000000)) | sudo -n tee $D/userspace/set_freq >/dev/null || { echo "set $f failed" >&2; continue; }
  sleep 1
  c0=$(cat $D/cur_freq); t0=$(maxtemp); a0=$(cat $D/device/power/runtime_active_time 2>/dev/null || echo 0)
  s=$(date +%s.%N)
  if [[ -n ${CPUS:-} ]]; then taskset -c "$CPUS" "$@" > "$OUT/${f}_r$r.log" 2>&1; else "$@" > "$OUT/${f}_r$r.log" 2>&1; fi; rc=$?
  e=$(date +%s.%N); a1=$(cat $D/device/power/runtime_active_time 2>/dev/null || echo 0)
  echo "$f,$r,$rc,$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.2f", b-a}'),$((a1 - a0)),$c0,$(cat $D/cur_freq),$t0,$(maxtemp),$OUT/${f}_r$r.log"
done
