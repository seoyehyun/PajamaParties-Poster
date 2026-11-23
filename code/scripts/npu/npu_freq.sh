#!/bin/bash
# usage: npu_freq.sh <model> <reps> <freq_mhz...>   (env NPU_INPUT_DIR: raw input<i>.bin files)   Measures NPU latency per OPP, restores 1200 MHz on exit.
D=/sys/class/devfreq/CIXH4000:00; model=$1; reps=$2; shift 2; inputs=${NPU_INPUT_DIR:-}
orig=$(cat $D/userspace/set_freq 2>/dev/null); echo "governor=$(cat $D/governor) set_freq_before=$orig cur_before=$(cat $D/cur_freq)"
restore(){ case "$orig" in ''|*[!0-9]*) v=1200000000;; *) v=$orig;; esac; echo $v | sudo -n tee $D/userspace/set_freq >/dev/null; echo "restored set_freq=$(cat $D/userspace/set_freq) cur=$(cat $D/cur_freq)"; }
trap restore EXIT
for f in "$@"; do
  echo $((f*1000000)) | sudo -n tee $D/userspace/set_freq >/dev/null || { echo "set $f failed"; exit 1; }
  sleep 1
  a0=$(cat $D/device/power/runtime_active_time)
  r=$(PYTHONPATH=$HOME/npu-models/.pydeps taskset -c 6-9 python3 -P "$(dirname "$0")/npu_smoke.py" "$model" "$reps" $inputs 2>&1 | grep '^{')
  a1=$(cat $D/device/power/runtime_active_time)
  t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  echo "f=$f set=$(cat $D/userspace/set_freq) cur=$(cat $D/cur_freq) active_ms=$((a1-a0)) maxtemp_mC=$t $r"
done
