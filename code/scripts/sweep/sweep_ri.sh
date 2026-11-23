#!/usr/bin/env bash
# Two-resource frequency grid sweep for the RI matrix. Run on the board. NOT yet run on hardware (see docs/sweep/freq-sweep.md).
# usage: scripts/sweep/sweep_ri.sh <mode>        mode is passed to the run command (base | file | select)
# env:   A, B          resources to sweep: cpu | gpu | npu | mem. B may be empty for a single-axis sweep
#        FREQS_A FREQS_B   MHz lists, e.g. FREQS_A="1008 1608 2304"
#        CPUS          cores for the cpu resource, default 6-9 (A720, policies 6 and 8: use <= 2200 MHz). A520: CPUS=2-5
#        GPU_DEV NPU_DEV MEM_DEV   devfreq node names under /sys/class/devfreq, auto-detected by pattern if unset
#        RUN_CMD       command that runs one measurement, default scripts/kvswap/run_stage1.sh (llama.cpp CPU build).
#                      It gets OUTDIR and must leave err.txt (llama_perf output) in it, or the metric columns stay empty.
#        REPS          repetitions per frequency combination, default 3
#        COOL_MC       wait until the hottest thermal zone is below this (milli-C) before each run, default 55000
#        FREQ_TOL      cpu freq_ok tolerance: allowed |real - requested| / requested, default 0.02. The real frequency is
#                      delta del / delta ref of the CPPC feedback counters (acpi_cppc/feedback_ctrs) of every core in CPUS.
#                      scaling_cur_freq only echoes the request: the hardware rounds a request up to its next step
#                      (docs/sweep/results/freq_steps_20261110.csv), so pick frequencies that are real steps of every policy in CPUS
#        OUTDIR        result root, default results/ri_<A>[_<B>]_<mode>_<date>
# The resource that is not swept is left as it is. Everything that was changed is restored on exit.
set -euo pipefail
mode=${1:?mode: base|file|select}
HERE=$(cd "$(dirname "$0")" && pwd)
A=${A:?A=cpu|gpu|npu}; B=${B:-}
FREQS_A=${FREQS_A:?FREQS_A="MHz list"}; FREQS_B=${FREQS_B:-}
CPUS=${CPUS:-6-9}; export CPUS   # Orange Pi 6 Plus: cpu6-9 = A720 on two cpufreq policies (max 2300 / 2200 MHz)
REPS=${REPS:-3}; COOL_MC=${COOL_MC:-55000}
RUN_CMD=${RUN_CMD:-$HERE/../kvswap/run_stage1.sh}
ROOT=${OUTDIR:-runs/ri_${A}${B:+_$B}_${mode}_$(date +%Y%m%d_%H%M)}; mkdir -p "$ROOT"
[[ -n $B && -z $FREQS_B ]] && { echo "FREQS_B required when B is set" >&2; exit 1; }
[[ $A == "$B" ]] && { echo "A and B must differ" >&2; exit 1; }

cpu_list() { local o=() p; IFS=, read -ra parts <<< "$1"
  for p in "${parts[@]}"; do if [[ $p == *-* ]]; then o+=($(seq "${p%-*}" "${p#*-}")); else o+=("$p"); fi; done; echo "${o[@]}"; }

# $1 name-pattern, $2 override, $3 kernel driver name (CIX boards name nodes CIXH5000:00, so match by driver too)
find_devfreq() {
  if [[ -n $2 ]]; then echo "/sys/class/devfreq/$2"; return; fi
  local d; for d in /sys/class/devfreq/*; do
    [[ $(basename "$d") == $1 ]] && { echo "$d"; return; }
    [[ -n ${3:-} && $(basename "$(readlink "$d/device/driver" 2>/dev/null)") == "$3" ]] && { echo "$d"; return; }
  done
  echo "no devfreq node matching $1 / driver ${3:-none}; set the *_DEV env var (ls /sys/class/devfreq)" >&2; return 1
}

# dirs of a resource; kHz/Hz unit and file names depend on the kind
res_dirs() {
  case $1 in
    cpu) local c d; declare -A s; for c in $(cpu_list "$CPUS"); do
           d=$(readlink -f /sys/devices/system/cpu/cpu$c/cpufreq)
           if [[ -d $d && -z ${s[$d]:-} ]]; then s[$d]=1; echo "$d"; fi; done ;;
    gpu) find_devfreq '*gpu*' "${GPU_DEV:-}" mali ;;      # CIXH5000:00
    npu) find_devfreq '*npu*' "${NPU_DEV:-}" armchina ;;  # CIXH4000:00
    mem) find_devfreq '*dmc*' "${MEM_DEV:-}" ;;          # none on the Orange Pi 6 Plus (see docs/sweep/freq-sweep-board-checks.md)
    *) echo "unknown resource $1" >&2; return 1 ;;
  esac
}
fmin() { [[ $1 == cpu ]] && echo scaling_min_freq || echo min_freq; }
fmax() { [[ $1 == cpu ]] && echo scaling_max_freq || echo max_freq; }
fgov() { [[ $1 == cpu ]] && echo scaling_governor || echo governor; }
fcur() { [[ $1 == cpu ]] && echo scaling_cur_freq || echo cur_freq; }
funit() { [[ $1 == cpu ]] && echo 1000 || echo 1000000; }   # MHz -> file unit

declare -A MINV MAXV GOVV CURV; RES=("$A"); [[ -n $B ]] && RES+=("$B")
declare -A DIRS
for r in "${RES[@]}"; do
  DIRS[$r]=$(res_dirs "$r" | tr '\n' ' ')
  [[ -n ${DIRS[$r]// } ]] || { echo "no frequency node for $r" >&2; exit 1; }
  echo "$r nodes: ${DIRS[$r]}"
  for d in ${DIRS[$r]}; do
    MINV[$d]=$(cat "$d/$(fmin "$r")"); MAXV[$d]=$(cat "$d/$(fmax "$r")")
    GOVV[$d]=$(cat "$d/$(fgov "$r")"); CURV[$d]=$(cat "$d/$(fcur "$r")")
  done
done
[[ ($A == mem || $B == mem) ]] && echo "note: the Orange Pi 6 Plus has no memory devfreq node, so mem will not be found" >&2

# every set_freq/restore uses sudo: ask once up front and keep the ticket alive for long sweeps
if ! sudo -n true 2>/dev/null; then sudo -v || exit 1; fi
( while kill -0 $$ 2>/dev/null; do sudo -n -v 2>/dev/null; sleep 50; done ) & SUDOKEEP=$!

restore() {
  local r d
  for r in "${RES[@]}"; do for d in ${DIRS[$r]}; do
    echo "${MAXV[$d]}" | sudo tee "$d/$(fmax "$r")" >/dev/null 2>&1 || true
    echo "${MINV[$d]}" | sudo tee "$d/$(fmin "$r")" >/dev/null 2>&1 || true
    echo "${GOVV[$d]}" | sudo tee "$d/$(fgov "$r")" >/dev/null 2>&1 || true
    # a devfreq node whose original governor is userspace keeps following set_freq: put the original frequency back
    if [[ -e $d/userspace/set_freq ]]; then echo "${CURV[$d]}" | sudo tee "$d/userspace/set_freq" >/dev/null 2>&1 || true; fi
  done; done
  kill ${TPID:-} ${SUDOKEEP:-} 2>/dev/null || true
}
trap restore EXIT

set_freq() { # resource MHz
  local r=$1 v=$(( $2 * $(funit "$1") )) d cur
  for d in ${DIRS[$r]}; do
    echo "${MAXV[$d]}" | sudo tee "$d/$(fmax "$r")" >/dev/null
    echo "${MINV[$d]}" | sudo tee "$d/$(fmin "$r")" >/dev/null
    # cpu: userspace + setspeed; devfreq: performance, then min=max pins it (works for NPU and GPU, checked in scripts/sweep/freq_check.sh)
    if [[ $r == cpu ]]; then echo userspace | sudo tee "$d/scaling_governor" >/dev/null
    else echo performance | sudo tee "$d/governor" >/dev/null; fi
    if (( v > MAXV[$d] )); then echo "$r $d: ${v} is above the limit ${MAXV[$d]} of this node" >&2; return 1; fi
    echo "$v" | sudo tee "$d/$(fmin "$r")" >/dev/null
    echo "$v" | sudo tee "$d/$(fmax "$r")" >/dev/null
    [[ $r == cpu ]] && echo "$v" | sudo tee "$d/scaling_setspeed" >/dev/null
    [[ -e $d/userspace/set_freq ]] && echo "$v" | sudo tee "$d/userspace/set_freq" >/dev/null   # NPU: userspace devfreq governor
  done
  sleep 1
  for d in ${DIRS[$r]}; do
    cur=$(cat "$d/$(fcur "$r")")
    [[ $cur -eq $v ]] || { echo "freq not applied: $r $d want $v got $cur" >&2; return 1; }
  done
}

maxtemp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
wait_cool() { local t; while t=$(maxtemp); [[ ${t:-0} -gt $COOL_MC ]]; do echo "  cooling: ${t} > ${COOL_MC}"; sleep 10; done; }
metric() { grep -E "$1" "$2" 2>/dev/null | grep -v "${3:-^$}" | grep -oE "$4" | head -1 | grep -oE '[0-9.]+' || true; }

sched=()
for fa in $FREQS_A; do
  if [[ -n $B ]]; then for fb in $FREQS_B; do for r in $(seq 1 "$REPS"); do sched+=("$fa:$fb:$r"); done; done
  else for r in $(seq 1 "$REPS"); do sched+=("$fa:-:$r"); done; fi
done
mapfile -t sched < <(printf '%s\n' "${sched[@]}" | shuf)

CSV=$ROOT/ri.csv
echo "mode,res_a,freq_a_mhz,res_b,freq_b_mhz,rep,exit,start_iso,end_iso,pp_tps,tg_tps,tpot_ms,temp_start_mC,temp_max_mC,temp_end_mC,majflt,max_rss_kb,freq_ok,freq_real_mhz,util_sel_p,util_sel_d,util_all_d,iowait_d,dir" > "$CSV"

for item in "${sched[@]}"; do
  IFS=: read -r fa fb r <<< "$item"
  dir=$ROOT/${A}${fa}${B:+_${B}${fb}}_r${r}; mkdir -p "$dir"
  echo "== $A ${fa} MHz ${B:+/ $B $fb MHz }rep $r =="
  wait_cool
  set_freq "$A" "$fa"; [[ -n $B ]] && set_freq "$B" "$fb"

  # frequency nodes to verify during the run, and the value each must show
  nodes=(); wants=()
  for pair in "$A:$fa" ${B:+"$B:$fb"}; do
    rr=${pair%%:*}; ff=${pair#*:}
    for d in ${DIRS[$rr]}; do nodes+=("$d/$(fcur "$rr")"); wants+=($(( ff * $(funit "$rr") ))); done
  done
  printf '%s\n' "${wants[@]}" > "$dir/freq.want"
  : > "$dir/temp.log"; : > "$dir/freq.log"; : > "$dir/util.log"; : > "$dir/ctr.log"
  CPUCORES=$(cpu_list "$CPUS")
  # once per second: temperature, actual frequencies, per-core CPU counters
  ( while :; do
      now=$(date +%s.%N)
      echo "$now $(maxtemp)" >> "$dir/temp.log"
      echo "$now $(cat "${nodes[@]}" | tr '\n' ' ')" >> "$dir/freq.log"
      { echo "T $now"; grep -E '^cpu[0-9]* ' /proc/stat; } >> "$dir/util.log"
      { echo "T $now"; for c in $CPUCORES; do echo "c$c $(cat /sys/devices/system/cpu/cpu$c/acpi_cppc/feedback_ctrs 2>/dev/null)"; done; } >> "$dir/ctr.log"
      sleep 1
    done ) & TPID=$!
  t0=$(date -Is); rc=0
  OUTDIR="$dir" "$RUN_CMD" "$mode" > "$dir/run.log" 2>&1 || rc=$?
  t1=$(date -Is); kill $TPID 2>/dev/null || true; TPID=

  e=$dir/err.txt
  pp=$(metric 'prompt eval time' "$e" '' '[0-9.]+ tokens per second')
  tg=$(metric ' eval time' "$e" 'prompt eval' '[0-9.]+ tokens per second')
  tpot=$(metric ' eval time' "$e" 'prompt eval' '[0-9.]+ ms per token')
  mf=$(metric 'Major \(requiring I/O\)' "$e" '' '[0-9]+$')
  rss=$(metric 'Maximum resident' "$e" '' '[0-9]+$')
  ts=$(grep temp_start_mC "$dir/meta.txt" 2>/dev/null | cut -d= -f2 || true)
  te=$(grep temp_end_mC "$dir/meta.txt" 2>/dev/null | cut -d= -f2 || true)
  tm=$(awk 'NF==2{if($2>m)m=$2}END{print m}' "$dir/temp.log")
  cpu_want=""; [[ $A == cpu ]] && cpu_want=$fa; [[ $B == cpu ]] && cpu_want=$fb
  post=$(python3 -I "$HERE/ri_post.py" "$dir" "$CPUS" "$cpu_want" "${FREQ_TOL:-0.02}" 2>"$dir/post.err") || post=",,,,,"
  [[ ${post%%,*} == 0 ]] && echo "  warning: frequency was not at the target during this run (freq_ok=0, real $(cut -d, -f2 <<< "$post") MHz), see $dir/freq.log and $dir/ctr.log" >&2
  echo "$mode,$A,$fa,$B,${fb/#-/},$r,$rc,$t0,$t1,$pp,$tg,$tpot,$ts,$tm,$te,$mf,$rss,$post,$dir" >> "$CSV"
  [[ $rc -ne 0 ]] && echo "  run failed (exit $rc), see $dir/run.log" >&2
done
echo "done: $CSV"
