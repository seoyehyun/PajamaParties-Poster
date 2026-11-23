#!/usr/bin/env bash
# KVSwap stage 1 single run (see docs/kvswap/kv-offload-plan.md). Run on the board.
# usage: scripts/kvswap/run_stage1.sh <mode>      mode: base | file | select
# env:   LLAMA_BIN MODEL PROMPT KV_FILE CTX NTOK OUTDIR
#        MEM_MAX   cgroup v2 memory limit, e.g. 2G (page cache is charged, swap is disabled)
#        M G W SINK PREFETCH   select-mode knobs (defaults 50 8 256 4 1)
#        ADAPTER   stage 2 predictor: adapter GGUF from scripts/kvswap/make_kvswap_adapter.py (select mode only; patch 04).
#                  PRED_EVAL=1 also logs attention-mass coverage (RAM-resident KV only), PRED_SAME=1 = no look-ahead
#        CPUS      cores to pin (taskset/mpstat list), default 6-7,8-9 (A720, cpufreq policies 6 and 8: use <= 2200 MHz)
#        THREADS   inference threads (-t), default 4; keep equal to the number of cores in CPUS
#        DEV       block device to watch in the summary, default sda (USB disk); recorded in meta.txt
#        KV_IO     direct (default) = explicit O_DIRECT reads/writes, KV is evicted from RAM after use, works WITHOUT a
#                  memory limit; mmap = stage-1 behaviour (page cache decides, needs MEM_MAX to force I/O)
#        IO_THREADS IO_DEPTH IO_CHUNK_KB   direct-I/O engine knobs (defaults 4, 1, 1024)
#        EXTRA     extra args for llama-completion
set -euo pipefail
mode=${1:?mode: base|file|select}
BIN=${LLAMA_BIN:-./build/bin/llama-completion}
MODEL=${MODEL:-$HOME/PajamaParties/models/Qwen3-1.7B-Q4_K_M.gguf}
PROMPT=${PROMPT:-prompt.txt}
KV_FILE=${KV_FILE:-$HOME/kv.bin}
CTX=${CTX:-32768}; NTOK=${NTOK:-1000}
CPUS=${CPUS:-6-7,8-9}; THREADS=${THREADS:-4}; DEV=${DEV:-sda}   # Orange Pi 6 Plus: cpu6-9 = A720 (2.3/2.2 GHz)
KV_IO=${KV_IO:-direct}
OUT=${OUTDIR:-.}; mkdir -p "$OUT"

case "$mode" in
  base) ;;
  file)   export LLAMA_KV_OFFLOAD_PATH=$KV_FILE ;;
  select) export LLAMA_KV_OFFLOAD_PATH=$KV_FILE LLAMA_KV_SELECT=${M:-50} LLAMA_KV_GROUP=${G:-8} \
                 LLAMA_KV_WINDOW=${W:-256} LLAMA_KV_SINK_GROUPS=${SINK:-4} LLAMA_KV_PREFETCH=${PREFETCH:-1} ;;
  *) echo "unknown mode $mode" >&2; exit 1 ;;
esac
if [[ "$mode" == select && -n "${ADAPTER:-}" ]]; then
  export LLAMA_KV_ADAPTER=$ADAPTER LLAMA_KV_PRED_EVAL=${PRED_EVAL:-0} LLAMA_KV_PRED_SAME=${PRED_SAME:-0}
fi
if [[ "$mode" != base && "$KV_IO" == direct ]]; then
  export LLAMA_KV_IO=direct LLAMA_KV_IO_THREADS=${IO_THREADS:-4} LLAMA_KV_IO_DEPTH=${IO_DEPTH:-1} LLAMA_KV_IO_CHUNK_KB=${IO_CHUNK_KB:-1024}
fi

temp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
{
  echo "mode=$mode"; echo "mem=${MEM_MAX:-none}"; echo "ctx=$CTX"; echo "ntok=$NTOK"; echo "io=$([[ "$mode" == base ]] && echo none || echo $KV_IO)"; echo "io_depth=${LLAMA_KV_IO_DEPTH:-}"; echo "io_threads=${LLAMA_KV_IO_THREADS:-}"; echo "cpus=$CPUS"; echo "threads=$THREADS"; echo "dev=$DEV"
  echo "G=${LLAMA_KV_GROUP:-}"; echo "M=${LLAMA_KV_SELECT:-}"; echo "W=${LLAMA_KV_WINDOW:-}"; echo "prefetch=${LLAMA_KV_PREFETCH:-}"; echo "adapter=${LLAMA_KV_ADAPTER:-}"
  echo "temp_start_mC=$(temp)"; echo "date=$(date -Is)"
} > "$OUT/meta.txt"

sync
# cold page cache; needs root. Without passwordless sudo it is skipped (direct-I/O mode does not depend on it)
if sudo -n true 2>/dev/null; then echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
else echo "warning: no passwordless sudo, page cache not dropped" >&2; fi
rm -f "$KV_FILE"

run=(taskset -c "$CPUS" "$BIN" -m "$MODEL" -f "$PROMPT" -c "$CTX" -n "$NTOK" --ignore-eos --temp 0 --seed 1 \
     -fa on -t "$THREADS" --no-warmup --no-display-prompt -lv 4 ${EXTRA:-})

cg=""
cleanup() { kill ${IOS:-} ${MPS:-} 2>/dev/null || true; [[ -n "$cg" ]] && sudo rmdir "$cg" 2>/dev/null || true; }
trap cleanup EXIT

if [[ -n "${MEM_MAX:-}" ]]; then
  if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then            # cgroup v2
    cg=/sys/fs/cgroup/kvswap_$$
    echo +memory | sudo tee /sys/fs/cgroup/cgroup.subtree_control >/dev/null 2>&1 || true
    sudo mkdir "$cg"
    echo "$MEM_MAX" | sudo tee "$cg/memory.max" >/dev/null
    # swapping to disk would confound the result; memory.swap.max is absent without swap accounting
    if [[ -f "$cg/memory.swap.max" ]]; then
      echo 0 | sudo tee "$cg/memory.swap.max" >/dev/null
    elif [[ $(wc -l < /proc/swaps) -gt 1 ]]; then
      echo "swap is active and $cg/memory.swap.max is missing; run 'sudo swapoff -a' first" >&2; exit 1
    fi
  else
    echo "no usable cgroup memory controller; refusing to run without the memory limit" >&2; exit 1
  fi
  # move the shell into the cgroup, then exec the run (env is kept with -E)
  run=(sudo -E sh -c "echo \$\$ > $cg/cgroup.procs; exec \"\$@\"" sh "${run[@]}")
fi

if command -v iostat >/dev/null; then iostat -x 1 > "$OUT/iostat.log" & IOS=$!; fi
if command -v mpstat >/dev/null; then mpstat -P "$CPUS" 1 > "$OUT/mpstat.log" & MPS=$!; fi
TIMEV=(); [[ -x /usr/bin/time ]] && TIMEV=(/usr/bin/time -v)
set +e
"${TIMEV[@]}" "${run[@]}" </dev/null > "$OUT/out.txt" 2> "$OUT/err.txt"
rc=$?
set -e
echo "exit=$rc" >> "$OUT/meta.txt"; echo "temp_end_mC=$(temp)" >> "$OUT/meta.txt"
[[ $rc -ne 0 ]] && echo "run failed (exit $rc), see $OUT/err.txt" >&2
grep -E "eval time|Major \(requiring I/O\)|Maximum resident" "$OUT/err.txt" || true
exit $rc
